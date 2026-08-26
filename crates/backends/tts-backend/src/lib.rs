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

/// Ports handed out to spawned `kokoro_tts_server.py` instances.
static NEXT_TTS_PORT: AtomicU16 = AtomicU16::new(59100);

/// Claim the next port nothing is listening on. The counter restarts at 59100
/// on every daemon start, so without this probe a new server can collide with
/// an orphaned one left behind by a previous run.
fn next_free_tts_port() -> u16 {
    for _ in 0..64 {
        let port = NEXT_TTS_PORT.fetch_add(1, Ordering::SeqCst);
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
        warn!("TTS port {} already in use (orphaned server?), trying the next one", port);
    }
    NEXT_TTS_PORT.fetch_add(1, Ordering::SeqCst)
}

/// How long to wait for the Python server to print "READY" on startup.
/// Model loading (Kokoro weights + espeak init) can take a while on first run.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

pub struct TtsBackend {
    model_path: Option<PathBuf>,
    is_loaded: Arc<AtomicBool>,
    supported_modalities: Vec<Modality>,
    process: Option<Child>,
    port: u16,
}

impl TtsBackend {
    pub fn new() -> Self {
        Self {
            model_path: None,
            is_loaded: Arc::new(AtomicBool::new(false)),
            supported_modalities: vec![Modality::AudioTts],
            process: None,
            port: 0,
        }
    }

    /// Resolve the Python interpreter used to run the TTS server scripts,
    /// preferring the project-local `.venv-tts` environment (built by
    /// `scripts/setup_tts_env.py`) that houses the heavy TTS-model deps system
    /// Python can't build. Resolution order:
    ///   1. `DISPOS_TTS_PYTHON` env var, if it points at an existing file.
    ///   2. `.venv-tts/Scripts/python.exe` (Windows) / `.venv-tts/bin/python`
    ///      (unix), relative to the current working directory.
    ///   3. Fallback to `"python"` on PATH.
    fn resolve_tts_python() -> PathBuf {
        if let Ok(custom) = std::env::var("DISPOS_TTS_PYTHON") {
            let path = PathBuf::from(custom);
            if path.exists() {
                return path;
            }
        }

        let venv_python = if cfg!(windows) {
            PathBuf::from(".venv-tts").join("Scripts").join("python.exe")
        } else {
            PathBuf::from(".venv-tts").join("bin").join("python")
        };
        if venv_python.exists() {
            return venv_python;
        }

        PathBuf::from("python")
    }

    /// Spawn `scripts/kokoro_tts_server.py` and block (on a worker thread) until
    /// it prints "READY" on stdout or `STARTUP_TIMEOUT` elapses.
    fn spawn_kokoro_server(model_path: &Path, port: u16) -> Result<Child, String> {
        let script_path = PathBuf::from("scripts").join("kokoro_tts_server.py");
        if !script_path.exists() {
            return Err(format!(
                "Kokoro TTS server script not found at: {}",
                script_path.display()
            ));
        }

        let mut cmd = Command::new(Self::resolve_tts_python());
        cmd.arg(&script_path)
            .arg("--model-path")
            .arg(model_path)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn kokoro_tts_server.py: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture stdout of kokoro_tts_server.py".to_string())?;

        let start = Instant::now();
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    return Err("kokoro_tts_server.py exited before signalling READY".to_string());
                }
                Ok(_) => {
                    if line.trim() == "READY" {
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("Error reading kokoro_tts_server.py stdout: {}", e));
                }
            }
            if start.elapsed() > STARTUP_TIMEOUT {
                let _ = child.kill();
                return Err("Timed out waiting for kokoro_tts_server.py to start".to_string());
            }
        }

        Ok(child)
    }

    /// Spawn `scripts/tts_server.py` (generic transformers-based TTS) and
    /// block (on a worker thread) until it prints "READY" on stdout or
    /// `STARTUP_TIMEOUT` elapses.
    fn spawn_generic_tts_server(model_path: &Path, port: u16) -> Result<Child, String> {
        let script_path = PathBuf::from("scripts").join("tts_server.py");
        if !script_path.exists() {
            return Err(format!(
                "Generic TTS server script not found at: {}",
                script_path.display()
            ));
        }

        let mut cmd = Command::new(Self::resolve_tts_python());
        cmd.arg(&script_path)
            .arg("--model-path")
            .arg(model_path)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn tts_server.py: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture stdout of tts_server.py".to_string())?;

        let start = Instant::now();
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    return Err("tts_server.py exited before signalling READY".to_string());
                }
                Ok(_) => {
                    if line.trim() == "READY" {
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("Error reading tts_server.py stdout: {}", e));
                }
            }
            if start.elapsed() > STARTUP_TIMEOUT {
                let _ = child.kill();
                return Err("Timed out waiting for tts_server.py to start".to_string());
            }
        }

        Ok(child)
    }

    /// Ensure the appropriate TTS server (`kokoro_tts_server.py` or
    /// `tts_server.py`, selected via `is_kokoro_model`) has been spawned for
    /// the currently loaded model. No-op if a process is already running.
    /// Returns `BackendError::EnvNotInstalled` if the `.venv-tts` environment
    /// hasn't been set up yet (rather than attempting to spawn against a
    /// missing interpreter/deps).
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

        let marker = PathBuf::from(".venv-tts").join(".install_complete");
        if !marker.exists() {
            return Err(BackendError::EnvNotInstalled("tts".to_string()));
        }

        let is_kokoro = is_kokoro_model(&model_path);
        let port = next_free_tts_port();
        let model_path_owned = model_path.clone();

        let spawn_result = if is_kokoro {
            tokio::task::spawn_blocking(move || Self::spawn_kokoro_server(&model_path_owned, port))
                .await
        } else {
            tokio::task::spawn_blocking(move || Self::spawn_generic_tts_server(&model_path_owned, port))
                .await
        };

        match spawn_result
            .map_err(|e| BackendError::LoadError(format!("spawn_blocking join error: {}", e)))?
        {
            Ok(child) => {
                let script_name = if is_kokoro { "kokoro_tts_server.py" } else { "tts_server.py" };
                info!("{} ready on port {} (PID: {})", script_name, port, child.id());
                self.process = Some(child);
                self.port = port;
            }
            Err(e) => {
                // The file exists but the server could not serve it (wrong format,
                // missing dependency, …). Report that instead of quietly serving
                // placeholder audio that looks like a success.
                self.process = None;
                self.is_loaded.store(false, Ordering::SeqCst);
                return Err(BackendError::LoadError(format!(
                    "Could not start the TTS server for '{}': {}",
                    model_path.display(),
                    e
                )));
            }
        }

        Ok(())
    }
}

/// Heuristic: does this model path point at a Kokoro (ONNX) model? Kokoro is
/// handled by its own special-cased script (custom G2P + voice-embedding
/// pipeline that doesn't generalize); everything else goes through the
/// generic transformers-based `tts_server.py`.
fn is_kokoro_model(model_path: &Path) -> bool {
    let path_str = model_path.to_string_lossy().to_lowercase();
    if path_str.contains("kokoro") {
        return true;
    }

    if model_path.is_file() {
        return model_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("onnx"))
            .unwrap_or(false);
    }

    if model_path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(model_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if name.ends_with(".onnx") && name.contains("kokoro") {
                    return true;
                }
            }
        }
    }

    false
}

impl Default for TtsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TtsBackend {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[async_trait]
impl InferenceBackend for TtsBackend {
    fn name(&self) -> &'static str {
        "kokoro-tts"
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
            BackendError::LoadError(format!("Failed to read file metadata for TTS VRAM estimate: {}", e))
        })?;

        let file_size = metadata.len();
        let estimated_vram = (file_size as f64 * 1.1) as u64;

        Ok(VramEstimate {
            required_bytes: estimated_vram,
            recommended_gpu_layers: 99,
            total_layers: 8,
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
                "TTS model path '{}' not found on disk; starting backend in simulation mode.",
                model_path.display()
            );
            self.model_path = Some(model_path.to_path_buf());
            self.is_loaded.store(true, Ordering::SeqCst);
            return Ok(());
        }

        let is_kokoro = is_kokoro_model(model_path);
        if is_kokoro {
            info!("Loading Kokoro TTS model: {} (server spawn deferred to first generate)", model_path.display());
        } else {
            info!(
                "Loading generic (transformers) TTS model: {} (server spawn deferred to first generate)",
                model_path.display()
            );
        }

        // Kill any previously running server before deciding on the new
        // model's mode.
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        self.model_path = Some(model_path.to_path_buf());
        self.is_loaded.store(true, Ordering::SeqCst);

        if is_kokoro {
            info!("Successfully loaded model in Kokoro TTS backend");
        } else {
            info!("Successfully loaded model in generic TTS backend");
        }
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<(), BackendError> {
        if self.is_loaded.load(Ordering::SeqCst) {
            info!("Unloading Kokoro TTS model from memory");
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

        let pcm_wav_bytes = if self.process.is_some() {
            let client = reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());

            let payload = serde_json::json!({
                "text": request.prompt,
                "voice": "af_heart",
                "speed": request.audio_params.as_ref().and_then(|a| a.speed).unwrap_or(1.0),
            });

            let url = format!("http://127.0.0.1:{}/synthesize", self.port);
            let response = client
                .post(&url)
                .json(&payload)
                .send()
                .await
                .map_err(|e| BackendError::InferenceError(format!("kokoro_tts_server connection error: {}", e)))?;

            if !response.status().is_success() {
                let status = response.status();
                let err_text = response.text().await.unwrap_or_default();
                return Err(BackendError::InferenceError(format!(
                    "kokoro_tts_server HTTP {} error: {}",
                    status, err_text
                )));
            }

            let body: serde_json::Value = response
                .json()
                .await
                .map_err(|e| BackendError::InferenceError(format!("kokoro_tts_server response parse error: {}", e)))?;

            let audio_b64 = body["audio_b64"]
                .as_str()
                .ok_or_else(|| BackendError::InferenceError("kokoro_tts_server response missing audio_b64".to_string()))?;

            base64::engine::general_purpose::STANDARD
                .decode(audio_b64)
                .map_err(|e| BackendError::InferenceError(format!("Failed to decode audio_b64: {}", e)))?
        } else {
            // Simulation mode: no Python server running (e.g. missing model file).
            vec![
                0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20,
                0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
                0x44, 0xAC, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00,
                0x02, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61,
            ]
        };

        let duration = start_time.elapsed().as_millis() as u64;

        Ok(InferenceResponse {
            request_id: request.request_id,
            output_text: "Audio synthesized successfully".to_string(),
            output_data: Some(pcm_wav_bytes),
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
            delta_text: "Synthesizing audio frames...".to_string(),
            delta_data: None,
            is_final: true,
            delta_tool_call: None,
        })];

        Ok(Box::pin(stream::iter(chunks)))
    }
}
