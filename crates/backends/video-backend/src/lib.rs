use std::path::{Path, PathBuf};
use async_trait::async_trait;
use backend_trait::{
    BackendError, GenerationProgress, InferenceBackend, InferenceChunk, InferenceRequest,
    InferenceResponse, InferenceStream, LoadOptions, Modality, VideoParams, VramEstimate,
};
use futures::stream;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use base64::Engine;

/// Minimal valid MP4 (empty `ftyp` box), used as the simulation-mode
/// fallback video when no `video_diffusers_server.py` process is running.
const SIMULATION_MP4: &[u8] = &[
    0x00, 0x00, 0x00, 0x1c, 0x66, 0x74, 0x79, 0x70,
    0x69, 0x73, 0x6f, 0x6d, 0x00, 0x00, 0x02, 0x00,
];

/// Ports handed out to spawned `video_diffusers_server.py` instances.
static NEXT_VIDEO_PORT: AtomicU16 = AtomicU16::new(59600);

/// Claim the next port nothing is listening on. The counter restarts at 59600
/// on every daemon start, so without this probe a new server can collide with
/// an orphaned one left behind by a previous run.
fn next_free_video_port() -> u16 {
    for _ in 0..64 {
        let port = NEXT_VIDEO_PORT.fetch_add(1, Ordering::SeqCst);
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
        warn!("Video port {} already in use (orphaned server?), trying the next one", port);
    }
    NEXT_VIDEO_PORT.fetch_add(1, Ordering::SeqCst)
}

/// How long to wait for `video_diffusers_server.py` to print "READY" on
/// startup. Video diffusion pipelines (and a possible first-time HF
/// download) can be slow to load.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(600);

pub struct VideoBackend {
    model_path: Option<PathBuf>,
    is_loaded: Arc<AtomicBool>,
    supported_modalities: Vec<Modality>,
    /// Set when `scripts/video_diffusers_server.py` was spawned successfully.
    /// Presence of this field selects the HTTP generation path.
    process: Option<Child>,
    port: u16,
}

impl VideoBackend {
    pub fn new() -> Self {
        Self {
            model_path: None,
            is_loaded: Arc::new(AtomicBool::new(false)),
            supported_modalities: vec![Modality::Video],
            process: None,
            port: 0,
        }
    }

    /// Spawn `scripts/video_diffusers_server.py` and block (on a worker
    /// thread) until it prints "READY" on stdout or `STARTUP_TIMEOUT` elapses.
    fn spawn_video_server(model_path: &Path, port: u16) -> Result<Child, String> {
        let script_path = PathBuf::from("scripts").join("video_diffusers_server.py");
        if !script_path.exists() {
            return Err(format!(
                "video_diffusers_server.py script not found at: {}",
                script_path.display()
            ));
        }

        let mut cmd = Command::new("python");
        cmd.arg(&script_path)
            .arg("--model-path")
            .arg(model_path)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn video_diffusers_server.py: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture stdout of video_diffusers_server.py".to_string())?;

        let start = Instant::now();
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    return Err("video_diffusers_server.py exited before signalling READY".to_string());
                }
                Ok(_) => {
                    if line.trim() == "READY" {
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("Error reading video_diffusers_server.py stdout: {}", e));
                }
            }
            if start.elapsed() > STARTUP_TIMEOUT {
                let _ = child.kill();
                return Err("Timed out waiting for video_diffusers_server.py to start".to_string());
            }
        }

        Ok(child)
    }

    /// POST the prompt+params to the running `video_diffusers_server.py`
    /// instance and decode the returned base64 MP4.
    async fn generate_video(
        port: u16,
        prompt: &str,
        params: &VideoParams,
    ) -> Result<Vec<u8>, BackendError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let image_b64 = params
            .image
            .as_ref()
            .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes));

        let mut payload = serde_json::json!({
            "prompt": prompt,
            "negative_prompt": params.negative_prompt,
            "image_b64": image_b64,
            "num_frames": params.num_frames,
            "height": params.height,
            "width": params.width,
            "num_inference_steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
            "fps": params.fps,
            "seed": params.seed,
        });
        // Pass any adapter-specific params straight through to
        // video_diffusers_server.py, untouched by the named fields above.
        if let serde_json::Value::Object(map) = &mut payload {
            map.insert("extra".to_string(), serde_json::to_value(&params.extra).unwrap_or_default());
        }

        let url = format!("http://127.0.0.1:{}/generate", port);
        let response = client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| BackendError::InferenceError(format!("video_diffusers_server connection error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let err_text = response.text().await.unwrap_or_default();
            return Err(BackendError::InferenceError(format!(
                "video_diffusers_server HTTP {} error: {}",
                status, err_text
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| BackendError::InferenceError(format!("video_diffusers_server response parse error: {}", e)))?;

        let video_b64 = body["video_b64"]
            .as_str()
            .ok_or_else(|| BackendError::InferenceError("video_diffusers_server response missing video_b64".to_string()))?;

        base64::engine::general_purpose::STANDARD
            .decode(video_b64)
            .map_err(|e| BackendError::InferenceError(format!("Failed to decode video_b64: {}", e)))
    }

    /// GET `/progress` from the running `video_diffusers_server.py` instance
    /// and map its JSON into a `GenerationProgress`. `job_id`/`modality` are
    /// left for daemon-core to stamp.
    async fn poll_video_progress(port: u16) -> Option<GenerationProgress> {
        let client = reqwest::Client::builder().no_proxy().build().ok()?;
        let url = format!("http://127.0.0.1:{}/progress", port);
        let response = client.get(&url).send().await.ok()?;
        if !response.status().is_success() {
            return None;
        }
        let body: serde_json::Value = response.json().await.ok()?;
        if body.get("status").and_then(|v| v.as_str()) == Some("idle") {
            return None;
        }

        let updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Some(GenerationProgress {
            job_id: String::new(),
            modality: "video".to_string(),
            phase: body.get("phase").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            step: body.get("step").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            total: body.get("total").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            percent: body.get("percent").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            status: body.get("status").and_then(|v| v.as_str()).unwrap_or("running").to_string(),
            message: body.get("message").and_then(|v| v.as_str()).map(|s| s.to_string()),
            media_handle: None,
            media_type: None,
            updated_at,
        })
    }
}

impl Default for VideoBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VideoBackend {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[async_trait]
impl InferenceBackend for VideoBackend {
    fn name(&self) -> &'static str {
        "video-diffusers"
    }

    fn supported_modalities(&self) -> &[Modality] {
        &self.supported_modalities
    }

    async fn estimate_vram(
        &self,
        model_path: &Path,
        _options: &LoadOptions,
    ) -> Result<VramEstimate, BackendError> {
        let metadata = std::fs::metadata(model_path).map_err(|e| {
            BackendError::LoadError(format!("Failed to read file metadata for Video VRAM estimate: {}", e))
        })?;

        let file_size = metadata.len();
        let estimated_vram = (file_size as f64 * 1.5) as u64;

        Ok(VramEstimate {
            required_bytes: estimated_vram,
            recommended_gpu_layers: 99,
            total_layers: 40,
            fits_in_vram: true,
        })
    }

    async fn load_model(
        &mut self,
        model_path: &Path,
        _options: &LoadOptions,
    ) -> Result<(), BackendError> {
        // Kill any previously running video server before deciding on the
        // new model's mode.
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        if !model_path.exists() {
            info!(
                "Video model path '{}' not found on disk; starting backend in simulation mode.",
                model_path.display()
            );
            self.model_path = Some(model_path.to_path_buf());
            self.is_loaded.store(true, Ordering::SeqCst);
            return Ok(());
        }

        info!("Loading diffusers video model: {}", model_path.display());
        let port = next_free_video_port();
        let model_path_owned = model_path.to_path_buf();

        // Unlike the "model path doesn't exist" branch above (an intentional
        // dev-mode simulation fallback), a spawn failure for a model that
        // genuinely exists on disk must be a hard error. Silently falling
        // back to simulation mode here previously left `is_loaded()`
        // reporting success while `/v1/videos/generations` quietly served
        // the tiny placeholder `SIMULATION_MP4` and `/v1/videos/schema`
        // 503'd — indistinguishable, from the UI's perspective, from a
        // genuinely broken video panel.
        let child = tokio::task::spawn_blocking(move || Self::spawn_video_server(&model_path_owned, port))
            .await
            .map_err(|e| BackendError::LoadError(format!("spawn_blocking join error: {}", e)))?
            .map_err(|e| {
                warn!("Could not start the video server for '{}': {}", model_path.display(), e);
                BackendError::LoadError(format!("Failed to start video_diffusers_server.py: {}", e))
            })?;

        info!("video_diffusers_server.py ready on port {} (PID: {})", port, child.id());
        self.process = Some(child);
        self.port = port;

        self.model_path = Some(model_path.to_path_buf());
        self.is_loaded.store(true, Ordering::SeqCst);

        info!("Successfully loaded model in Video backend");
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<(), BackendError> {
        if self.is_loaded.load(Ordering::SeqCst) {
            info!("Unloading Video model from memory");
            if let Some(mut child) = self.process.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
            self.model_path = None;
            self.is_loaded.store(false, Ordering::SeqCst);
        }
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.is_loaded.load(Ordering::SeqCst)
    }

    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::ModelNotLoaded);
        }

        let start_time = std::time::Instant::now();
        let params = request.video_params.clone().unwrap_or_default();

        info!("Executing video generation prompt: '{}'", request.prompt);

        let mp4_bytes: Vec<u8> = if self.process.is_some() {
            Self::generate_video(self.port, &request.prompt, &params).await?
        } else {
            // Simulation mode: no video_diffusers_server running (e.g.
            // missing model file or missing runtime dependencies).
            SIMULATION_MP4.to_vec()
        };

        let duration = start_time.elapsed().as_millis() as u64;

        Ok(InferenceResponse {
            request_id: request.request_id,
            output_text: "Video generation completed successfully".to_string(),
            output_data: Some(mp4_bytes),
            tokens_generated: 1,
            generation_time_ms: duration,
            tool_calls: None,
            finish_reason: None,
        })
    }

    async fn generate_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceStream, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::ModelNotLoaded);
        }

        let req_id = request.request_id.clone();
        let chunks = vec![Ok(InferenceChunk {
            request_id: req_id,
            delta_text: "Generating video...".to_string(),
            delta_data: None,
            is_final: true,
            delta_tool_call: None,
        })];

        Ok(Box::pin(stream::iter(chunks)))
    }

    async fn poll_progress(&self) -> Option<GenerationProgress> {
        if self.process.is_none() {
            // No video_diffusers_server.py running (simulation mode) — no
            // progress source available.
            return None;
        }
        Self::poll_video_progress(self.port).await
    }

    async fn get_param_schema(&self) -> Option<serde_json::Value> {
        if self.process.is_none() {
            // No video_diffusers_server.py running (simulation mode) — no
            // schema source.
            return None;
        }
        let client = reqwest::Client::builder().no_proxy().build().ok()?;
        let url = format!("http://127.0.0.1:{}/schema", self.port);
        let response = client.get(&url).send().await.ok()?;
        if !response.status().is_success() {
            return None;
        }
        response.json().await.ok()
    }
}
