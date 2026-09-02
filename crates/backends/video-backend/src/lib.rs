use std::path::{Path, PathBuf};
use async_trait::async_trait;
use backend_trait::{
    BackendError, GenerationProgress, InferenceBackend, InferenceChunk, InferenceRequest,
    InferenceResponse, InferenceStream, LoadOptions, Modality, VideoParams, VramEstimate,
};
use futures::stream;
use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tracing::{info, warn};
use base64::Engine;

/// Snapshot of the GGUF de-quantization progress reported by `transformers`'
/// tqdm bar on `video_diffusers_server.py`'s stderr during first-time model
/// load. Polled by `GET /v1/model/load-progress` in daemon-core.
#[derive(Clone, Debug, serde::Serialize)]
pub struct LoadProgressSnapshot {
    pub phase: String,
    pub percent: f32,
    pub current: u64,
    pub total: u64,
}

/// Process-wide slot for the currently in-flight model load's progress, if
/// any. Set by the stderr-reader thread spawned in `spawn_video_server`;
/// cleared when that function returns (success or failure).
static LOAD_PROGRESS: OnceLock<Mutex<Option<LoadProgressSnapshot>>> = OnceLock::new();

fn load_progress_slot() -> &'static Mutex<Option<LoadProgressSnapshot>> {
    LOAD_PROGRESS.get_or_init(|| Mutex::new(None))
}

/// Read the current model-load progress snapshot, if a load is in progress
/// and has emitted at least one parseable tqdm line.
pub fn current_load_progress() -> Option<LoadProgressSnapshot> {
    load_progress_slot().lock().ok().and_then(|guard| guard.clone())
}

fn set_load_progress(snapshot: LoadProgressSnapshot) {
    if let Ok(mut guard) = load_progress_slot().lock() {
        *guard = Some(snapshot);
    }
}

fn clear_load_progress() {
    if let Ok(mut guard) = load_progress_slot().lock() {
        *guard = None;
    }
}

/// Parse one tqdm progress line, e.g.:
/// `"Converting and de-quantizing GGUF tensors...:   4%| | 10/242 [00:02<01:20, 2.8it/s]"`
/// into `(phase, percent, current, total)`.
fn parse_tqdm_line(line: &str) -> Option<(String, f32, u64, u64)> {
    let line = line.trim();
    let colon_idx = line.find(':')?;
    let phase = line[..colon_idx].trim().to_string();
    let rest = &line[colon_idx + 1..];
    let pct_idx = rest.find('%')?;
    let percent: f32 = rest[..pct_idx].trim().parse().ok()?;
    let frac_tok = rest.split_whitespace().find(|tok| {
        let core = tok.trim_matches(|c: char| !c.is_ascii_digit() && c != '/');
        let mut parts = core.split('/');
        matches!((parts.next(), parts.next(), parts.next()),
            (Some(a), Some(b), None) if !a.is_empty() && !b.is_empty()
                && a.chars().all(|c| c.is_ascii_digit())
                && b.chars().all(|c| c.is_ascii_digit()))
    })?;
    let core = frac_tok.trim_matches(|c: char| !c.is_ascii_digit() && c != '/');
    let mut parts = core.split('/');
    let current: u64 = parts.next()?.parse().ok()?;
    let total: u64 = parts.next()?.parse().ok()?;
    Some((phase, percent, current, total))
}

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

/// HF repo LTX-Video's missing pipeline components (for loose single-file
/// transformer+VAE loading) are pulled from.
const LTX_BASE_REPO: &str = "Lightricks/LTX-Video";

/// Files needed under `<shared components dir>/text_encoder/`.
const LTX_TEXT_ENCODER_FILES: &[&str] = &[
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
];
/// Files needed under `<shared components dir>/tokenizer/`.
const LTX_TOKENIZER_FILES: &[&str] = &["added_tokens.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"];
/// Files needed under `<shared components dir>/scheduler/`.
const LTX_SCHEDULER_FILES: &[&str] = &["scheduler_config.json"];

/// Resolution status of one LTX-Video pipeline component file (text encoder
/// shard, tokenizer file, scheduler config), for preview in the model
/// list/config UI (see `VideoBackend::detect_video_components`). Mirrors
/// `sd_backend::ImageComponentStatus`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VideoComponentStatus {
    /// UI grouping label, e.g. "Text encoder" / "Tokenizer" / "Scheduler".
    pub group: String,
    pub kind_name: String,
    /// Set when an existing file satisfying this component was found.
    pub resolved_path: Option<String>,
    /// Set when nothing was found: the directory it should be placed in.
    pub target_path: Option<String>,
    /// Set when nothing was found and a known download source exists.
    pub source: Option<backend_trait::MissingComponentSource>,
}

pub struct VideoBackend {
    model_path: Option<PathBuf>,
    is_loaded: Arc<AtomicBool>,
    supported_modalities: Vec<Modality>,
    /// Set when `scripts/video_diffusers_server.py` was spawned successfully.
    /// Presence of this field selects the HTTP generation path.
    process: Option<Child>,
    port: u16,
    /// Manual override for the LTX-Video text encoder (gguf/safetensors/ckpt/pt/bin).
    /// `None` falls back to sibling auto-detection.
    text_encoder_override: Option<PathBuf>,
    /// Manual override for the LTX-Video VAE (gguf/safetensors/ckpt/pt/bin).
    /// `None` falls back to sibling auto-detection.
    vae_override: Option<PathBuf>,
}

impl VideoBackend {
    pub fn new() -> Self {
        Self {
            model_path: None,
            is_loaded: Arc::new(AtomicBool::new(false)),
            supported_modalities: vec![Modality::Video],
            process: None,
            port: 0,
            text_encoder_override: None,
            vae_override: None,
        }
    }

    /// Resolve the Python interpreter used to run `scripts/video_diffusers_server.py`,
    /// preferring the project-local `.venv-video` environment (built by
    /// `scripts/setup_video_env.py`) that houses the heavy video-model deps system
    /// Python can't build. Resolution order:
    ///   1. `DISPOS_VIDEO_PYTHON` env var, if it points at an existing file.
    ///   2. `.venv-video/Scripts/python.exe` (Windows) / `.venv-video/bin/python`
    ///      (unix), relative to the current working directory.
    ///   3. Fallback to `"python"` on PATH.
    fn resolve_video_python() -> PathBuf {
        if let Ok(custom) = std::env::var("DISPOS_VIDEO_PYTHON") {
            let path = PathBuf::from(custom);
            if path.exists() {
                return path;
            }
        }

        let venv_python = if cfg!(windows) {
            PathBuf::from(".venv-video").join("Scripts").join("python.exe")
        } else {
            PathBuf::from(".venv-video").join("bin").join("python")
        };
        if venv_python.exists() {
            return venv_python;
        }

        PathBuf::from("python")
    }

    /// Locate DisposAI's `models` directory the same way `daemon-core` and
    /// `sd-backend` do (`DISPOS_MODELS_DIR` override, then walking up from
    /// the current/exe directory, then the workspace source tree for dev
    /// builds).
    fn resolve_models_root() -> Option<PathBuf> {
        let mut candidates = Vec::new();

        if let Ok(directory) = std::env::var("DISPOS_MODELS_DIR") {
            candidates.push(PathBuf::from(directory));
        }
        if let Ok(current_dir) = std::env::current_dir() {
            candidates.extend(current_dir.ancestors().map(|path| path.join("models")));
        }
        if let Ok(executable) = std::env::current_exe() {
            if let Some(parent) = executable.parent() {
                candidates.extend(parent.ancestors().map(|path| path.join("models")));
            }
        }
        candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models"));

        candidates.into_iter().find(|directory| directory.is_dir())
    }

    /// Directory where downloaded LTX-Video pipeline components (text
    /// encoder/tokenizer/scheduler) are cached, shared across all loose LTX
    /// checkpoints.
    fn ltx_shared_components_dir(models_root: &Path) -> PathBuf {
        models_root.join("_shared_components").join("ltx-video")
    }

    /// Preview the LTX-Video pipeline components (text encoder, tokenizer,
    /// scheduler) a loose single-file transformer+VAE checkpoint needs,
    /// without erroring on anything missing — used by the model list/config
    /// UI to surface resolved paths (and downloadable sources for anything
    /// absent) before the user attempts generation. Returns `None` for
    /// non-LTX models or full pipeline directories, which don't need this.
    pub fn detect_video_components(model_path: &Path) -> Option<Vec<VideoComponentStatus>> {
        if model_path.is_dir() {
            return None;
        }
        let filename = model_path.file_name()?.to_str()?.to_ascii_lowercase();
        if !filename.contains("ltx") {
            return None;
        }
        let ext = model_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        if ext != "gguf" && ext != "safetensors" {
            return None;
        }

        let models_root = Self::resolve_models_root()?;
        let components_dir = Self::ltx_shared_components_dir(&models_root);

        let mut statuses = Vec::new();

        // A single quantized `.gguf` text encoder file (any quant level)
        // satisfies the text encoder leg on its own, in place of the full
        // fp16 shard set below (see `_find_text_encoder_gguf` in
        // video_diffusers_server.py).
        let text_encoder_dir = components_dir.join("text_encoder");
        let local_gguf = std::fs::read_dir(&text_encoder_dir).ok().and_then(|entries| {
            entries.filter_map(|e| e.ok()).find_map(|e| {
                let path = e.path();
                let is_gguf = path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false);
                (path.is_file() && is_gguf).then_some(path)
            })
        });
        if let Some(gguf_path) = local_gguf {
            statuses.push(VideoComponentStatus {
                group: "Text encoder".to_string(),
                kind_name: gguf_path.file_name().and_then(|n| n.to_str()).unwrap_or("text_encoder.gguf").to_string(),
                resolved_path: Some(gguf_path.display().to_string()),
                target_path: Some(gguf_path.display().to_string()),
                source: None,
            });
        } else {
            for file in LTX_TEXT_ENCODER_FILES {
                let local_path = text_encoder_dir.join(file);
                if local_path.is_file() {
                    statuses.push(VideoComponentStatus {
                        group: "Text encoder".to_string(),
                        kind_name: (*file).to_string(),
                        resolved_path: Some(local_path.display().to_string()),
                        target_path: Some(local_path.display().to_string()),
                        source: None,
                    });
                } else {
                    statuses.push(VideoComponentStatus {
                        group: "Text encoder".to_string(),
                        kind_name: (*file).to_string(),
                        resolved_path: None,
                        target_path: Some(text_encoder_dir.display().to_string()),
                        source: Some(backend_trait::MissingComponentSource {
                            repo: LTX_BASE_REPO.to_string(),
                            filename: format!("text_encoder/{file}"),
                            target_filename: (*file).to_string(),
                        }),
                    });
                }
            }
        }

        let groups: [(&str, &str, &[&str]); 2] = [
            ("Tokenizer", "tokenizer", LTX_TOKENIZER_FILES),
            ("Scheduler", "scheduler", LTX_SCHEDULER_FILES),
        ];

        for (group, subdir, files) in groups {
            let subdir_path = components_dir.join(subdir);
            for file in files {
                let local_path = subdir_path.join(file);
                if local_path.is_file() {
                    statuses.push(VideoComponentStatus {
                        group: group.to_string(),
                        kind_name: (*file).to_string(),
                        resolved_path: Some(local_path.display().to_string()),
                        target_path: Some(local_path.display().to_string()),
                        source: None,
                    });
                } else {
                    statuses.push(VideoComponentStatus {
                        group: group.to_string(),
                        kind_name: (*file).to_string(),
                        resolved_path: None,
                        target_path: Some(subdir_path.display().to_string()),
                        source: Some(backend_trait::MissingComponentSource {
                            repo: LTX_BASE_REPO.to_string(),
                            filename: format!("{subdir}/{file}"),
                            target_filename: (*file).to_string(),
                        }),
                    });
                }
            }
        }
        Some(statuses)
    }

    /// Spawn `scripts/video_diffusers_server.py` and block (on a worker
    /// thread) until it prints "READY" on stdout or `STARTUP_TIMEOUT` elapses.
    fn spawn_video_server(
        model_path: &Path,
        port: u16,
        text_encoder_override: Option<&Path>,
        vae_override: Option<&Path>,
    ) -> Result<Child, String> {
        let script_path = PathBuf::from("scripts").join("video_diffusers_server.py");
        if !script_path.exists() {
            return Err(format!(
                "video_diffusers_server.py script not found at: {}",
                script_path.display()
            ));
        }

        let mut cmd = Command::new(Self::resolve_video_python());
        cmd.arg(&script_path)
            .arg("--model-path")
            .arg(model_path)
            .arg("--port")
            .arg(port.to_string());
        // Tell the script where to look for locally-downloaded LTX pipeline
        // components before it falls back to auto-downloading them from HF.
        // Harmless to pass for non-LTX models; the script only consults it
        // when loading a loose LTX transformer file.
        if let Some(models_root) = Self::resolve_models_root() {
            cmd.arg("--ltx-components-dir").arg(Self::ltx_shared_components_dir(&models_root));
        }
        // Manual overrides for the LTX-Video text encoder/VAE, bypassing
        // sibling auto-detection when provided.
        if let Some(path) = text_encoder_override {
            cmd.arg("--text-encoder-path").arg(path);
        }
        if let Some(path) = vae_override {
            cmd.arg("--vae-path").arg(path);
        }
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn video_diffusers_server.py: {}", e))?;

        clear_load_progress();

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture stdout of video_diffusers_server.py".to_string())?;

        // tqdm overwrites its progress line in place with `\r`, only emitting
        // a real `\n` once complete, so intermediate updates would be
        // invisible to a `read_line`-based reader. Read raw bytes instead and
        // split on both `\r` and `\n`, forwarding each segment to the real
        // stderr for terminal visibility while also updating LOAD_PROGRESS
        // when a segment parses as a tqdm line.
        if let Some(mut stderr) = child.stderr.take() {
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    match stderr.read(&mut chunk) {
                        Ok(0) => break,
                        Ok(n) => {
                            for &byte in &chunk[..n] {
                                if byte == b'\r' || byte == b'\n' {
                                    if !buf.is_empty() {
                                        let segment = String::from_utf8_lossy(&buf).to_string();
                                        eprintln!("{}", segment);
                                        if let Some((phase, percent, current, total)) = parse_tqdm_line(&segment) {
                                            set_load_progress(LoadProgressSnapshot { phase, percent, current, total });
                                        }
                                        buf.clear();
                                    }
                                } else {
                                    buf.push(byte);
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
                if !buf.is_empty() {
                    eprintln!("{}", String::from_utf8_lossy(&buf));
                }
            });
        }

        let start = Instant::now();
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    clear_load_progress();
                    return Err("video_diffusers_server.py exited before signalling READY".to_string());
                }
                Ok(_) => {
                    if line.trim() == "READY" {
                        break;
                    }
                }
                Err(e) => {
                    clear_load_progress();
                    return Err(format!("Error reading video_diffusers_server.py stdout: {}", e));
                }
            }
            if start.elapsed() > STARTUP_TIMEOUT {
                let _ = child.kill();
                clear_load_progress();
                return Err("Timed out waiting for video_diffusers_server.py to start".to_string());
            }
        }

        clear_load_progress();
        Ok(child)
    }

    /// POST the prompt+params to the running `video_diffusers_server.py`
    /// instance and decode the returned base64 MP4.
    /// POST /cancel to the running `video_diffusers_server.py`, best-effort.
    /// The server is threaded so this can be handled while /generate is
    /// still in flight on another thread; it sets a flag the pipeline's
    /// step callback checks to abort generation early.
    async fn cancel_video_server(port: u16) {
        let client = match reqwest::Client::builder().no_proxy().build() {
            Ok(c) => c,
            Err(_) => return,
        };
        let url = format!("http://127.0.0.1:{}/cancel", port);
        let _ = client.post(&url).send().await;
    }

    async fn generate_video(
        port: u16,
        prompt: &str,
        params: &VideoParams,
        cancel: Option<&Arc<AtomicBool>>,
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
        let request_fut = client.post(&url).json(&payload).send();

        let response = if let Some(cancel) = cancel {
            let cancel = cancel.clone();
            let cancel_wait = async {
                loop {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(150)).await;
                }
            };
            tokio::select! {
                result = request_fut => {
                    result.map_err(|e| BackendError::InferenceError(format!("video_diffusers_server connection error: {}", e)))?
                }
                _ = cancel_wait => {
                    Self::cancel_video_server(port).await;
                    return Err(BackendError::Cancelled);
                }
            }
        } else {
            request_fut
                .await
                .map_err(|e| BackendError::InferenceError(format!("video_diffusers_server connection error: {}", e)))?
        };

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

    /// Ensure `video_diffusers_server.py` has been spawned for the currently
    /// loaded model. No-op if a process is already running. Returns
    /// `BackendError::EnvNotInstalled` if the `.venv-video` environment
    /// hasn't been set up yet (rather than attempting to spawn against a
    /// missing interpreter/deps). Unlike the "model path doesn't exist" case
    /// in `load_model` (an intentional dev-mode simulation fallback), a spawn
    /// failure for a model that genuinely exists on disk is a hard error:
    /// silently falling back to simulation mode here would leave
    /// `is_loaded()` reporting success while `/v1/videos/generations`
    /// quietly served the tiny placeholder `SIMULATION_MP4` and
    /// `/v1/videos/schema` 503'd — indistinguishable, from the UI's
    /// perspective, from a genuinely broken video panel.
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

        let marker = PathBuf::from(".venv-video").join(".install_complete");
        if !marker.exists() {
            return Err(BackendError::EnvNotInstalled("video".to_string()));
        }

        info!("Starting video_diffusers_server.py for: {}", model_path.display());
        let port = next_free_video_port();
        let model_path_owned = model_path.clone();
        let text_encoder_override_owned = self.text_encoder_override.clone();
        let vae_override_owned = self.vae_override.clone();

        let child = tokio::task::spawn_blocking(move || {
            Self::spawn_video_server(
                &model_path_owned,
                port,
                text_encoder_override_owned.as_deref(),
                vae_override_owned.as_deref(),
            )
        })
            .await
            .map_err(|e| BackendError::LoadError(format!("spawn_blocking join error: {}", e)))?
            .map_err(|e| {
                warn!("Could not start the video server for '{}': {}", model_path.display(), e);
                BackendError::LoadError(format!("Failed to start video_diffusers_server.py: {}", e))
            })?;

        info!("video_diffusers_server.py ready on port {} (PID: {})", port, child.id());
        self.process = Some(child);
        self.port = port;

        Ok(())
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
        options: &LoadOptions,
    ) -> Result<(), BackendError> {
        // Kill any previously running video server before deciding on the
        // new model's mode.
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        self.text_encoder_override = options
            .text_encoder_override_path
            .as_ref()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        self.vae_override = options
            .vae_override_path
            .as_ref()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        if !model_path.exists() {
            info!(
                "Video model path '{}' not found on disk; starting backend in simulation mode.",
                model_path.display()
            );
            self.model_path = Some(model_path.to_path_buf());
            self.is_loaded.store(true, Ordering::SeqCst);
            return Ok(());
        }

        info!(
            "Loading diffusers video model: {} (server spawn deferred to first generate)",
            model_path.display()
        );
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
        &mut self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::ModelNotLoaded);
        }

        self.ensure_process_started().await?;

        let start_time = std::time::Instant::now();
        let params = request.video_params.clone().unwrap_or_default();

        info!("Executing video generation prompt: '{}'", request.prompt);

        let mp4_bytes: Vec<u8> = if self.process.is_some() {
            Self::generate_video(self.port, &request.prompt, &params, request.cancel.as_ref()).await?
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

    async fn get_param_schema(&mut self) -> Option<serde_json::Value> {
        // Schema is static per-pipeline (no GPU load happens until an actual
        // `generate()` call), so it's safe to spawn video_diffusers_server.py
        // here rather than waiting for the user to trigger a generation first.
        let _ = self.ensure_process_started().await;
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

#[cfg(test)]
mod tests {
    use super::parse_tqdm_line;

    #[test]
    fn parses_gguf_dequantize_tqdm_line() {
        let line = "Converting and de-quantizing GGUF tensors...:   4%| | 10/242 [00:02<01:20, 2.8it/s]";
        let parsed = parse_tqdm_line(line).expect("should parse tqdm line");
        assert_eq!(parsed, ("Converting and de-quantizing GGUF tensors...".to_string(), 4.0, 10, 242));
    }
}
