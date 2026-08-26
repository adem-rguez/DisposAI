use std::path::{Path, PathBuf};
use async_trait::async_trait;
use backend_trait::{
    BackendError, GenerationProgress, InferenceBackend, InferenceChunk, InferenceRequest, InferenceResponse,
    InferenceStream, LoadOptions, Mesh3dParams, Modality, VramEstimate,
};
use futures::stream;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use base64::Engine;

/// Ports handed out to spawned `threed_server.py` instances.
static NEXT_MESH_PORT: AtomicU16 = AtomicU16::new(59400);

/// Claim the next port nothing is listening on. The counter restarts at 59400
/// on every daemon start, so without this probe a new server can collide with
/// an orphaned one left behind by a previous run.
fn next_free_mesh_port() -> u16 {
    for _ in 0..64 {
        let port = NEXT_MESH_PORT.fetch_add(1, Ordering::SeqCst);
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
        warn!("Mesh3D port {} already in use (orphaned server?), trying the next one", port);
    }
    NEXT_MESH_PORT.fetch_add(1, Ordering::SeqCst)
}

/// How long to wait for `threed_server.py` to print "READY" on startup. 3D
/// generation deps and models can take a while to download/load on first run.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(600);

/// Minimal valid ASCII OBJ unit cube, used as the simulation-mode fallback
/// mesh when no `threed_server.py` process is running.
const SIMULATION_CUBE_OBJ: &str = "\
# DisposAI simulation-mode placeholder mesh
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
v 0.0 1.0 1.0
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 4 8 5 1
";

pub struct MeshBackend {
    model_path: Option<PathBuf>,
    is_loaded: Arc<AtomicBool>,
    supported_modalities: Vec<Modality>,
    /// Set when `scripts/threed_server.py` was spawned successfully. Presence
    /// of this field selects the HTTP generation path.
    process: Option<Child>,
    port: u16,
}

impl MeshBackend {
    pub fn new() -> Self {
        Self {
            model_path: None,
            is_loaded: Arc::new(AtomicBool::new(false)),
            supported_modalities: vec![Modality::Mesh3D],
            process: None,
            port: 0,
        }
    }

    /// Resolve the Python interpreter used to run `scripts/threed_server.py`,
    /// preferring the project-local `.venv-3d` environment (built by
    /// `scripts/setup_3d_env.py`) that houses the heavy 3D-model deps system
    /// Python can't build. Resolution order:
    ///   1. `DISPOS_3D_PYTHON` env var, if it points at an existing file.
    ///   2. `.venv-3d/Scripts/python.exe` (Windows) / `.venv-3d/bin/python`
    ///      (unix), relative to the current working directory.
    ///   3. Fallback to `"python"` on PATH.
    fn resolve_3d_python() -> PathBuf {
        if let Ok(custom) = std::env::var("DISPOS_3D_PYTHON") {
            let path = PathBuf::from(custom);
            if path.exists() {
                return path;
            }
        }

        let venv_python = if cfg!(windows) {
            PathBuf::from(".venv-3d").join("Scripts").join("python.exe")
        } else {
            PathBuf::from(".venv-3d").join("bin").join("python")
        };
        if venv_python.exists() {
            return venv_python;
        }

        PathBuf::from("python")
    }

    /// Spawn `scripts/threed_server.py` and block (on a worker thread) until
    /// it prints "READY" on stdout or `STARTUP_TIMEOUT` elapses.
    fn spawn_threed_server(model_path: &Path, port: u16) -> Result<Child, String> {
        let script_path = PathBuf::from("scripts").join("threed_server.py");
        if !script_path.exists() {
            return Err(format!(
                "threed_server.py script not found at: {}",
                script_path.display()
            ));
        }

        let mut cmd = Command::new(Self::resolve_3d_python());
        cmd.arg(&script_path)
            .arg("--model-path")
            .arg(model_path)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn threed_server.py: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture stdout of threed_server.py".to_string())?;

        let start = Instant::now();
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    return Err("threed_server.py exited before signalling READY".to_string());
                }
                Ok(_) => {
                    if line.trim() == "READY" {
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("Error reading threed_server.py stdout: {}", e));
                }
            }
            if start.elapsed() > STARTUP_TIMEOUT {
                let _ = child.kill();
                return Err("Timed out waiting for threed_server.py to start".to_string());
            }
        }

        Ok(child)
    }

    /// POST the prompt/images+params to the running `threed_server.py`
    /// instance and decode the returned base64 mesh.
    async fn generate_mesh(
        port: u16,
        prompt: &str,
        params: &Mesh3dParams,
    ) -> Result<(Vec<u8>, String), BackendError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let images_b64: Vec<String> = params
            .images
            .clone()
            .unwrap_or_default()
            .iter()
            .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes))
            .collect();

        let mut payload = serde_json::json!({
            "prompt": prompt,
            "images": images_b64,
            "input_kind": params.input_kind,
            "steps": params.steps,
            "guidance_scale": params.guidance_scale,
            "seed": params.seed,
            "output_format": params.output_format,
            "texture": params.texture,
            "foreground_ratio": params.foreground_ratio,
        });
        // Pass any adapter-specific params (e.g. SF3D's `remesh_option`) straight
        // through to threed_server.py, untouched by the named fields above.
        if let serde_json::Value::Object(map) = &mut payload {
            for (key, value) in params.extra.iter() {
                map.insert(key.clone(), value.clone());
            }
        }

        let url = format!("http://127.0.0.1:{}/generate", port);
        let response = client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| BackendError::InferenceError(format!("threed_server connection error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let err_text = response.text().await.unwrap_or_default();
            return Err(BackendError::InferenceError(format!(
                "threed_server HTTP {} error: {}",
                status, err_text
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| BackendError::InferenceError(format!("threed_server response parse error: {}", e)))?;

        let mesh_b64 = body["mesh_base64"]
            .as_str()
            .ok_or_else(|| BackendError::InferenceError("threed_server response missing mesh_base64".to_string()))?;
        let format = body["format"].as_str().unwrap_or("glb").to_string();

        let bytes = base64::engine::general_purpose::STANDARD
            .decode(mesh_b64)
            .map_err(|e| BackendError::InferenceError(format!("Failed to decode mesh_base64: {}", e)))?;

        Ok((bytes, format))
    }

    /// GET `/progress` from the running `threed_server.py` instance and map
    /// its JSON into a `GenerationProgress`. `job_id`/`modality` are left
    /// for daemon-core to stamp.
    async fn poll_threed_progress(port: u16) -> Option<GenerationProgress> {
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
            modality: "mesh".to_string(),
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

    /// Ensure `threed_server.py` has been spawned for the currently loaded
    /// model. No-op if a process is already running. Returns
    /// `BackendError::EnvNotInstalled` if the `.venv-3d` environment hasn't
    /// been set up yet (rather than attempting to spawn against a missing
    /// interpreter/deps).
    async fn ensure_process_started(&mut self) -> Result<(), BackendError> {
        if self.process.is_some() {
            return Ok(());
        }

        let model_path = self
            .model_path
            .clone()
            .ok_or(BackendError::ModelNotLoaded)?;

        if !model_path.exists() {
            // Simulation mode: no real model file to serve, nothing to spawn.
            return Ok(());
        }

        let marker = PathBuf::from(".venv-3d").join(".install_complete");
        if !marker.exists() {
            return Err(BackendError::EnvNotInstalled("mesh3d".to_string()));
        }

        info!("Starting threed_server.py for: {}", model_path.display());
        let port = next_free_mesh_port();
        let model_path_owned = model_path.clone();

        match tokio::task::spawn_blocking(move || Self::spawn_threed_server(&model_path_owned, port))
            .await
            .map_err(|e| BackendError::LoadError(format!("spawn_blocking join error: {}", e)))?
        {
            Ok(child) => {
                info!("threed_server.py ready on port {} (PID: {})", port, child.id());
                self.process = Some(child);
                self.port = port;
            }
            Err(e) => {
                warn!(
                    "Could not start the 3D mesh server for '{}': {}. Falling back to simulation mode.",
                    model_path.display(),
                    e
                );
            }
        }

        Ok(())
    }
}

impl Default for MeshBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MeshBackend {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[async_trait]
impl InferenceBackend for MeshBackend {
    fn name(&self) -> &'static str {
        "mesh3d-generator"
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
            BackendError::LoadError(format!("Failed to read file metadata for Mesh3D VRAM estimate: {}", e))
        })?;

        let file_size = metadata.len();
        // Heuristic: similar order of magnitude to SD VRAM allocation.
        let estimated_vram = (file_size as f64 * 1.3) as u64;

        Ok(VramEstimate {
            required_bytes: estimated_vram,
            recommended_gpu_layers: 99,
            total_layers: 16,
            fits_in_vram: true,
        })
    }

    async fn load_model(
        &mut self,
        model_path: &Path,
        _options: &LoadOptions,
    ) -> Result<(), BackendError> {
        // Kill any previously running threed server before deciding on the
        // new model's mode.
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        if !model_path.exists() {
            info!(
                "Mesh3D model path '{}' not found on disk; starting backend in simulation mode.",
                model_path.display()
            );
            self.model_path = Some(model_path.to_path_buf());
            self.is_loaded.store(true, Ordering::SeqCst);
            return Ok(());
        }

        info!("Loading 3D mesh generator model: {}", model_path.display());
        self.model_path = Some(model_path.to_path_buf());
        self.is_loaded.store(true, Ordering::SeqCst);

        info!("Successfully loaded model in Mesh3D backend (server spawn deferred to first generate)");
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<(), BackendError> {
        if self.is_loaded.load(Ordering::SeqCst) {
            info!("Unloading Mesh3D model from memory");
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
        &mut self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::ModelNotLoaded);
        }

        self.ensure_process_started().await?;

        let start_time = std::time::Instant::now();
        let params = request.mesh_params.clone().unwrap_or_default();

        info!("Executing 3D mesh generation prompt: '{}'", request.prompt);

        let (mesh_bytes, format): (Vec<u8>, String) = if self.process.is_some() {
            Self::generate_mesh(self.port, &request.prompt, &params).await?
        } else {
            // Simulation mode: no threed_server running (e.g. missing model
            // file or missing runtime dependencies). Never error here.
            (SIMULATION_CUBE_OBJ.as_bytes().to_vec(), "obj".to_string())
        };

        let duration = start_time.elapsed().as_millis() as u64;

        Ok(InferenceResponse {
            request_id: request.request_id,
            output_text: format,
            output_data: Some(mesh_bytes),
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
            delta_text: "Generating 3D mesh...".to_string(),
            delta_data: None,
            is_final: true,
            delta_tool_call: None,
        })];

        Ok(Box::pin(stream::iter(chunks)))
    }

    async fn poll_progress(&self) -> Option<GenerationProgress> {
        if self.process.is_none() {
            // No threed_server.py running (simulation mode) — no progress
            // source available.
            return None;
        }
        Self::poll_threed_progress(self.port).await
    }

    async fn get_param_schema(&mut self) -> Option<serde_json::Value> {
        // Schema is static per-adapter (no GPU load happens until an actual
        // `generate()` call), so it's safe to spawn threed_server.py here
        // rather than waiting for the user to trigger a generation first.
        let _ = self.ensure_process_started().await;
        if self.process.is_none() {
            // No threed_server.py running (simulation mode) — no schema source.
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
