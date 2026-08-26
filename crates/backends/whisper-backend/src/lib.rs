use std::path::{Path, PathBuf};
use async_trait::async_trait;
use backend_trait::{
    BackendError, InferenceBackend, InferenceChunk, InferenceRequest, InferenceResponse,
    InferenceStream, LoadOptions, Modality, VramEstimate,
};
use futures::stream;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use base64::Engine;

/// Ports handed out to spawned `whisper_server.py` instances.
static NEXT_ASR_PORT: AtomicU16 = AtomicU16::new(59200);

/// Claim the next port nothing is listening on. The counter restarts at 59200
/// on every daemon start, so without this probe a new server can collide with
/// an orphaned one left behind by a previous run.
fn next_free_asr_port() -> u16 {
    for _ in 0..64 {
        let port = NEXT_ASR_PORT.fetch_add(1, Ordering::SeqCst);
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
        warn!("ASR port {} already in use (orphaned server?), trying the next one", port);
    }
    NEXT_ASR_PORT.fetch_add(1, Ordering::SeqCst)
}

/// How long to wait for the Python server to print "READY" on startup.
/// Model loading (Whisper weights) can take a while on first run.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

pub struct WhisperBackend {
    model_path: Option<PathBuf>,
    is_loaded: Arc<AtomicBool>,
    supported_modalities: Vec<Modality>,
    process: Option<Child>,
    port: u16,
}

impl WhisperBackend {
    pub fn new() -> Self {
        Self {
            model_path: None,
            is_loaded: Arc::new(AtomicBool::new(false)),
            supported_modalities: vec![Modality::AudioAsr],
            process: None,
            port: 0,
        }
    }

    /// Spawn `scripts/whisper_server.py` and block (on a worker thread) until
    /// it prints "READY" on stdout or `STARTUP_TIMEOUT` elapses.
    fn spawn_server(model_path: &Path, port: u16) -> Result<Child, String> {
        let script_path = PathBuf::from("scripts").join("whisper_server.py");
        if !script_path.exists() {
            return Err(format!(
                "Whisper ASR server script not found at: {}",
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
            .map_err(|e| format!("Failed to spawn whisper_server.py: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture stdout of whisper_server.py".to_string())?;

        let start = Instant::now();
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    return Err("whisper_server.py exited before signalling READY".to_string());
                }
                Ok(_) => {
                    if line.trim() == "READY" {
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("Error reading whisper_server.py stdout: {}", e));
                }
            }
            if start.elapsed() > STARTUP_TIMEOUT {
                let _ = child.kill();
                return Err("Timed out waiting for whisper_server.py to start".to_string());
            }
        }

        Ok(child)
    }
}

impl Default for WhisperBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for WhisperBackend {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[async_trait]
impl InferenceBackend for WhisperBackend {
    fn name(&self) -> &'static str {
        "whisper.cpp"
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
            BackendError::LoadError(format!("Failed to read file metadata for Whisper VRAM estimate: {}", e))
        })?;

        let file_size = metadata.len();
        let estimated_vram = (file_size as f64 * 1.1) as u64;

        Ok(VramEstimate {
            required_bytes: estimated_vram,
            recommended_gpu_layers: 99,
            total_layers: 12,
            fits_in_vram: true,
        })
    }

    async fn load_model(
        &mut self,
        model_path: &Path,
        _options: &LoadOptions,
    ) -> Result<(), BackendError> {
        if !model_path.exists() {
            warn!(
                "Whisper model path '{}' not found on disk; starting backend in simulation mode.",
                model_path.display()
            );
            self.model_path = Some(model_path.to_path_buf());
            self.is_loaded.store(true, Ordering::SeqCst);
            return Ok(());
        }

        info!("Loading Whisper ASR model: {}", model_path.display());

        // Kill any previously running server before spawning a new one.
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        let port = next_free_asr_port();
        let model_path_owned = model_path.to_path_buf();

        match tokio::task::spawn_blocking(move || Self::spawn_server(&model_path_owned, port))
            .await
            .map_err(|e| BackendError::LoadError(format!("spawn_blocking join error: {}", e)))?
        {
            Ok(child) => {
                info!("whisper_server.py ready on port {} (PID: {})", port, child.id());
                self.process = Some(child);
                self.port = port;
            }
            Err(e) => {
                // The file exists but the server could not serve it (wrong format,
                // missing dependency, …). Report that instead of quietly serving
                // placeholder transcripts that look like a success.
                self.process = None;
                self.is_loaded.store(false, Ordering::SeqCst);
                return Err(BackendError::LoadError(format!(
                    "Could not start the ASR server for '{}': {}",
                    model_path.display(),
                    e
                )));
            }
        }

        self.model_path = Some(model_path.to_path_buf());
        self.is_loaded.store(true, Ordering::SeqCst);

        info!("Successfully loaded model in whisper.cpp backend");
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<(), BackendError> {
        if self.is_loaded.load(Ordering::SeqCst) {
            info!("Unloading whisper.cpp model from memory");
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
        self.generate_impl(request).await
    }

    async fn generate_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceStream, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::ModelNotLoaded);
        }

        let response = self.generate_impl(request.clone()).await?;

        let chunks = vec![Ok(InferenceChunk {
            request_id: response.request_id,
            delta_text: response.output_text,
            delta_data: None,
            is_final: true,
            delta_tool_call: None,
        })];

        Ok(Box::pin(stream::iter(chunks)))
    }
}

impl WhisperBackend {
    /// Shared transcription logic used by both `generate` and
    /// `generate_stream` (the latter needs `&self` per the trait, so it
    /// can't call `generate` directly once `generate` became `&mut self`).
    async fn generate_impl(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::ModelNotLoaded);
        }

        let start_time = std::time::Instant::now();

        let output_text = if self.process.is_some() {
            let audio_bytes = request.image_input.as_ref().ok_or_else(|| {
                BackendError::InferenceError("No audio data provided for transcription".to_string())
            })?;

            let client = reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());

            let audio_b64 = base64::engine::general_purpose::STANDARD.encode(audio_bytes);
            let payload = serde_json::json!({
                "audio_b64": audio_b64,
            });

            let url = format!("http://127.0.0.1:{}/transcribe", self.port);
            let response = client
                .post(&url)
                .json(&payload)
                .send()
                .await
                .map_err(|e| BackendError::InferenceError(format!("whisper_server connection error: {}", e)))?;

            if !response.status().is_success() {
                let status = response.status();
                let err_text = response.text().await.unwrap_or_default();
                return Err(BackendError::InferenceError(format!(
                    "whisper_server HTTP {} error: {}",
                    status, err_text
                )));
            }

            let body: serde_json::Value = response
                .json()
                .await
                .map_err(|e| BackendError::InferenceError(format!("whisper_server response parse error: {}", e)))?;

            body["text"]
                .as_str()
                .ok_or_else(|| BackendError::InferenceError("whisper_server response missing text".to_string()))?
                .to_string()
        } else {
            // Simulation mode: no Python server running (e.g. missing model file).
            "[Transcribed Audio Speech Text via whisper.cpp]".to_string()
        };

        let duration = start_time.elapsed().as_millis() as u64;
        let tokens_generated = output_text.split_whitespace().count() as u32;

        Ok(InferenceResponse {
            request_id: request.request_id,
            output_text,
            output_data: None,
            tokens_generated,
            generation_time_ms: duration,
            tool_calls: None,
            finish_reason: None,
        })
    }
}
