mod component_specs;

use std::path::{Path, PathBuf};
use async_trait::async_trait;
use backend_trait::{
    BackendError, GenerationProgress, ImageParams, InferenceBackend, InferenceChunk, InferenceRequest,
    InferenceResponse, InferenceStream, LoadOptions, Modality, VramEstimate,
};
use component_specs::{match_component_spec, ComponentKind, ComponentSpec};
use futures::stream;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{info, warn};
use base64::Engine;

/// The specific stderr substring `sd.exe` emits when a `.gguf` file is a
/// standalone diffusion-transformer (no VAE/text-encoder) and can't be
/// loaded via the legacy full-checkpoint `-m` flag. Its presence is the
/// trigger to retry generation in split-checkpoint "component mode".
const SD_VERSION_SNIFF_FAILURE: &str = "get sd version from file failed";

/// Resolved arguments for one `sd.exe` invocation: either the legacy
/// single-file checkpoint path, or a split-checkpoint "component mode"
/// invocation built from a matched `ComponentSpec`.
enum SdCommandMode {
    Legacy(PathBuf),
    Component {
        diffusion_model: PathBuf,
        clip_l: Option<PathBuf>,
        clip_g: Option<PathBuf>,
        t5xxl: Option<PathBuf>,
        llm: Option<PathBuf>,
        vae: PathBuf,
        vae_format: Option<&'static str>,
        model_args: Option<&'static str>,
    },
}

/// Live progress reported by the `sd.exe` (GGUF) generation path, parsed
/// from the child process's stderr.
struct GgufProgress {
    active: bool,
    step: u32,
    total: u32,
    phase: String,
}

/// Ports handed out to spawned `diffusers_server.py` instances.
static NEXT_SD_PORT: AtomicU16 = AtomicU16::new(59200);

/// Claim the next port nothing is listening on. The counter restarts at 59200
/// on every daemon start, so without this probe a new server can collide with
/// an orphaned one left behind by a previous run.
fn next_free_sd_port() -> u16 {
    for _ in 0..64 {
        let port = NEXT_SD_PORT.fetch_add(1, Ordering::SeqCst);
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
        warn!("SD port {} already in use (orphaned server?), trying the next one", port);
    }
    NEXT_SD_PORT.fetch_add(1, Ordering::SeqCst)
}

/// How long to wait for `diffusers_server.py` to print "READY" on startup.
/// Diffusion pipelines (and a possible first-time HF download) can take a
/// while to load.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

/// Resolution status of one split-checkpoint sibling component, for preview
/// in the model list/config UI (see `SdBackend::detect_image_components`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ImageComponentStatus {
    pub kind_name: String,
    /// Set when an existing file satisfying this component was found.
    pub resolved_path: Option<String>,
    /// Set when nothing was found: the shared directory it should be placed in.
    pub target_path: Option<String>,
    /// Set when nothing was found and a known download source exists.
    pub source: Option<backend_trait::MissingComponentSource>,
}

pub struct SdBackend {
    model_path: Option<PathBuf>,
    is_loaded: Arc<AtomicBool>,
    supported_modalities: Vec<Modality>,
    /// Set when the loaded model is a `.gguf` file and `sd.exe` was found on
    /// disk. Presence of this field selects the one-shot CLI generation path.
    sd_binary: Option<PathBuf>,
    /// Set when the loaded model is a `.safetensors` file / HF-repo directory
    /// and `scripts/diffusers_server.py` was spawned successfully. Presence
    /// of this field selects the HTTP generation path.
    process: Option<Child>,
    port: u16,
    /// Live progress for the `sd.exe` (GGUF) generation path.
    gguf_progress: Arc<Mutex<GgufProgress>>,
}

impl SdBackend {
    pub fn new() -> Self {
        Self {
            model_path: None,
            is_loaded: Arc::new(AtomicBool::new(false)),
            supported_modalities: vec![Modality::Image],
            sd_binary: None,
            process: None,
            port: 0,
            gguf_progress: Arc::new(Mutex::new(GgufProgress {
                active: false,
                step: 0,
                total: 0,
                phase: String::new(),
            })),
        }
    }

    /// Locate the `stable-diffusion.cpp` CLI executable. Checked in order:
    /// an explicit `DISPOS_SD_BINARY` env var override, known install
    /// locations (mirrors `llama-backend`'s `find_llama_server_binary`), then
    /// the system PATH.
    fn find_sd_binary() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("DISPOS_SD_BINARY") {
            let path = PathBuf::from(p);
            if path.exists() {
                return Some(path);
            }
        }

        if let Ok(userprofile) = std::env::var("USERPROFILE") {
            let backends_dir = PathBuf::from(&userprofile).join(".lmstudio\\extensions\\backends");

            let direct = backends_dir.join("sd.exe");
            if direct.exists() {
                return Some(direct);
            }

            if let Ok(entries) = std::fs::read_dir(&backends_dir) {
                for entry in entries.flatten() {
                    let candidate = entry.path().join("sd.exe");
                    if candidate.exists() {
                        return Some(candidate);
                    }
                }
            }
        }

        #[cfg(windows)]
        {
            if let Ok(output) = Command::new("where").arg("sd.exe").output() {
                if output.status.success() {
                    if let Some(first_line) = String::from_utf8_lossy(&output.stdout).lines().next() {
                        let path = PathBuf::from(first_line.trim());
                        if path.exists() {
                            return Some(path);
                        }
                    }
                }
            }
        }
        #[cfg(not(windows))]
        {
            if let Ok(output) = Command::new("which").arg("sd").output() {
                if output.status.success() {
                    if let Some(first_line) = String::from_utf8_lossy(&output.stdout).lines().next() {
                        let path = PathBuf::from(first_line.trim());
                        if path.exists() {
                            return Some(path);
                        }
                    }
                }
            }
        }

        None
    }

    /// Spawn `scripts/diffusers_server.py` and block (on a worker thread)
    /// until it prints "READY" on stdout or `STARTUP_TIMEOUT` elapses.
    fn spawn_diffusers_server(model_path: &Path, port: u16) -> Result<Child, String> {
        let script_path = PathBuf::from("scripts").join("diffusers_server.py");
        if !script_path.exists() {
            return Err(format!(
                "diffusers_server.py script not found at: {}",
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
            .map_err(|e| format!("Failed to spawn diffusers_server.py: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture stdout of diffusers_server.py".to_string())?;

        let start = Instant::now();
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    return Err("diffusers_server.py exited before signalling READY".to_string());
                }
                Ok(_) => {
                    if line.trim() == "READY" {
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("Error reading diffusers_server.py stdout: {}", e));
                }
            }
            if start.elapsed() > STARTUP_TIMEOUT {
                let _ = child.kill();
                return Err("Timed out waiting for diffusers_server.py to start".to_string());
            }
        }

        Ok(child)
    }

    /// Parse a whitespace-delimited `N/M` step token (e.g. from a
    /// `  |====>  | 3/20 - 1.50it/s` sd.exe progress line) out of a chunk of
    /// stdout text. sd.exe's stdout carries several different `N/M`-shaped
    /// progress bars (the real per-step sampling counter, but also
    /// tensor-loading/VAE-decode counters and plain log lines like
    /// `generating image: 1/1 - seed 42`), so a match is only accepted as
    /// the real step counter if the token immediately following it looks
    /// like a per-iteration rate (contains `it/s` or ends with `s/it`,
    /// case-insensitive) rather than a throughput rate (`b/s`) or something
    /// else entirely. Returns `Some((step, total))` for the last such valid
    /// token found in the chunk.
    fn parse_step_total(line: &str) -> Option<(u32, u32)> {
        let mut found = None;
        let mut tokens = line.split_whitespace().peekable();
        while let Some(tok) = tokens.next() {
            if let Some((a, b)) = tok.split_once('/') {
                if let (Ok(step), Ok(total)) = (a.parse::<u32>(), b.parse::<u32>()) {
                    if total > 0 {
                        if let Some(next) = tokens.peek() {
                            let next_lower = next.to_ascii_lowercase();
                            let is_iter_rate = next_lower.contains("it/s") || next_lower.ends_with("s/it");
                            let is_throughput_rate = next_lower.ends_with("b/s");
                            if is_iter_rate && !is_throughput_rate {
                                found = Some((step, total));
                            }
                        }
                    }
                }
            }
        }
        found
    }

    /// Locate DisposAI's `models` directory the same way `daemon-core` does
    /// (`DISPOS_MODELS_DIR` override, then walking up from the current/exe
    /// directory, then the workspace source tree for dev builds).
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

    /// Search a single directory (non-recursively) for a file whose name
    /// contains one of `patterns` (case-insensitive).
    fn find_component_in_dir(dir: &Path, patterns: &[&str]) -> Option<PathBuf> {
        let entries = std::fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = path.file_name()?.to_str()?.to_ascii_lowercase();
            if patterns.iter().any(|p| name.contains(&p.to_ascii_lowercase())) {
                return Some(path);
            }
        }
        None
    }

    /// Search a directory tree (recursively) for a file whose name contains
    /// one of `patterns` (case-insensitive).
    fn find_component_recursive(dir: &Path, patterns: &[&str]) -> Option<PathBuf> {
        if let Some(found) = Self::find_component_in_dir(dir, patterns) {
            return Some(found);
        }
        let entries = std::fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = Self::find_component_recursive(&path, patterns) {
                    return Some(found);
                }
            }
        }
        None
    }

    /// Resolve one required component: first next to the diffusion model
    /// (tier 1), then in the shared components directory (tier 2).
    fn resolve_component(
        kind: &ComponentKind,
        diffusion_dir: &Path,
        models_root: Option<&Path>,
    ) -> Option<PathBuf> {
        let patterns = kind.filename_patterns();
        if let Some(found) = Self::find_component_in_dir(diffusion_dir, &patterns) {
            return Some(found);
        }
        let models_root = models_root?;
        let shared_dir = models_root.join(kind.shared_subdir().replace('/', std::path::MAIN_SEPARATOR_STR));
        Self::find_component_recursive(&shared_dir, &patterns)
    }

    /// Preview the split-checkpoint sibling components (text encoder, VAE)
    /// a standalone diffusion-model GGUF/checkpoint needs, without erroring
    /// on anything missing — used by the model list/config UI to surface
    /// resolved paths (and downloadable sources for anything absent) before
    /// the user attempts generation. Returns `None` if the filename doesn't
    /// match a known split-checkpoint architecture family (e.g. it's a
    /// self-contained single-file checkpoint).
    pub fn detect_image_components(model_path: &Path) -> Option<Vec<ImageComponentStatus>> {
        let filename = model_path.file_name().and_then(|n| n.to_str())?;
        let spec = match_component_spec(filename)?;
        let diffusion_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
        let models_root = Self::resolve_models_root();

        let statuses = spec
            .components
            .iter()
            .map(|kind| match Self::resolve_component(kind, diffusion_dir, models_root.as_deref()) {
                Some(path) => ImageComponentStatus {
                    kind_name: kind.display_name().to_string(),
                    resolved_path: Some(path.display().to_string()),
                    target_path: None,
                    source: None,
                },
                None => {
                    let shared_dir = models_root
                        .as_ref()
                        .map(|root| root.join(kind.shared_subdir().replace('/', std::path::MAIN_SEPARATOR_STR)))
                        .unwrap_or_else(|| PathBuf::from(format!("models/{}", kind.shared_subdir())));
                    let source = spec.default_source_for(kind).map(|src| backend_trait::MissingComponentSource {
                        repo: src.repo.to_string(),
                        filename: src.filename.to_string(),
                        target_filename: src.target_filename.to_string(),
                    });
                    ImageComponentStatus {
                        kind_name: kind.display_name().to_string(),
                        resolved_path: None,
                        target_path: Some(shared_dir.display().to_string()),
                        source,
                    }
                }
            })
            .collect();
        Some(statuses)
    }

    /// Build a `SdCommandMode::Component` for `model_path` by matching its
    /// filename against the architecture family table and resolving every
    /// required component. Returns the spec's `default_cfg_scale`/
    /// `default_steps` alongside the mode for `generate_gguf` to apply.
    fn resolve_component_mode(
        model_path: &Path,
    ) -> Result<(SdCommandMode, Option<f32>, Option<u32>), BackendError> {
        let filename = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        let spec: &'static ComponentSpec = match_component_spec(filename).ok_or_else(|| {
            BackendError::InferenceError(format!(
                "'{}' is a standalone diffusion-model file, but its architecture family \
                 isn't recognized for split-checkpoint (component-mode) loading. Add a \
                 matching pattern to sd-backend's component_specs table.",
                filename
            ))
        })?;

        let diffusion_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
        let models_root = Self::resolve_models_root();

        let mut clip_l = None;
        let mut clip_g = None;
        let mut t5xxl = None;
        let mut llm = None;
        let mut vae = None;
        let mut missing: Vec<backend_trait::MissingComponentInfo> = Vec::new();

        for kind in spec.components {
            let resolved = Self::resolve_component(kind, diffusion_dir, models_root.as_deref());
            match resolved {
                Some(path) => match kind {
                    ComponentKind::ClipL => clip_l = Some(path),
                    ComponentKind::ClipG => clip_g = Some(path),
                    ComponentKind::T5xxl => t5xxl = Some(path),
                    ComponentKind::Llm(_) => llm = Some(path),
                    ComponentKind::Vae(_) => vae = Some(path),
                },
                None => {
                    let shared_dir = models_root
                        .as_ref()
                        .map(|root| root.join(kind.shared_subdir().replace('/', std::path::MAIN_SEPARATOR_STR)))
                        .unwrap_or_else(|| PathBuf::from(format!("models/{}", kind.shared_subdir())));
                    let source = spec.default_source_for(kind).map(|src| {
                        backend_trait::MissingComponentSource {
                            repo: src.repo.to_string(),
                            filename: src.filename.to_string(),
                            target_filename: src.target_filename.to_string(),
                        }
                    });
                    let target_path = shared_dir.display().to_string();
                    missing.push(backend_trait::MissingComponentInfo {
                        kind_name: kind.display_name().to_string(),
                        target_path,
                        source,
                    });
                }
            }
        }

        if !missing.is_empty() {
            return Err(BackendError::MissingComponents(missing));
        }

        let vae = vae.ok_or_else(|| {
            BackendError::InferenceError(format!(
                "'{}' ({} family) requires a VAE component, but none was resolved.",
                filename, spec.family
            ))
        })?;

        Ok((
            SdCommandMode::Component {
                diffusion_model: model_path.to_path_buf(),
                clip_l,
                clip_g,
                t5xxl,
                llm,
                vae,
                vae_format: spec.vae_format,
                model_args: spec.model_args,
            },
            spec.default_cfg_scale,
            spec.default_steps,
        ))
    }

    /// Run `sd.exe` once with the given command mode, tracking live progress
    /// via `gguf_progress`, and return the resulting PNG bytes (or the raw
    /// error, uninspected, for the caller to decide whether to retry).
    async fn run_sd_exe(
        sd_binary: &Path,
        mode: &SdCommandMode,
        request_id: &str,
        prompt: &str,
        params: &ImageParams,
        cfg_scale_override: Option<f32>,
        steps_override: Option<u32>,
        gguf_progress: &Arc<Mutex<GgufProgress>>,
        cancel: Option<&Arc<AtomicBool>>,
        init_image: Option<&[u8]>,
    ) -> Result<Vec<u8>, BackendError> {
        let out_path = std::env::temp_dir().join(format!("dispos_sd_{}.png", request_id));
        let init_img_path = std::env::temp_dir().join(format!("dispos_sd_init_{}.png", request_id));
        if let Some(bytes) = init_image {
            if let Err(e) = tokio::fs::write(&init_img_path, bytes).await {
                return Err(BackendError::InferenceError(format!(
                    "Failed to write init image temp file: {}",
                    e
                )));
            }
        }
        let total_steps = params.steps.or(steps_override).unwrap_or(20);
        let cfg_scale = params.cfg_scale.or(cfg_scale_override).unwrap_or(7.0);

        let mut cmd = tokio::process::Command::new(sd_binary);
        match mode {
            SdCommandMode::Legacy(model_path) => {
                cmd.arg("-m").arg(model_path);
            }
            SdCommandMode::Component {
                diffusion_model,
                clip_l,
                clip_g,
                t5xxl,
                llm,
                vae,
                vae_format,
                model_args,
            } => {
                cmd.arg("--diffusion-model").arg(diffusion_model);
                if let Some(p) = clip_l {
                    cmd.arg("--clip_l").arg(p);
                }
                if let Some(p) = clip_g {
                    cmd.arg("--clip_g").arg(p);
                }
                if let Some(p) = t5xxl {
                    cmd.arg("--t5xxl").arg(p);
                }
                if let Some(p) = llm {
                    cmd.arg("--llm").arg(p);
                }
                cmd.arg("--vae").arg(vae);
                if let Some(format) = vae_format {
                    cmd.arg("--vae-format").arg(format);
                }
                if let Some(args) = model_args {
                    cmd.arg("--model-args").arg(args);
                }
            }
        }
        cmd.arg("-p").arg(prompt);
        if let Some(negative) = &params.negative_prompt {
            cmd.arg("-n").arg(negative);
        }
        cmd.arg("--steps").arg(total_steps.to_string());
        cmd.arg("--cfg-scale").arg(cfg_scale.to_string());
        cmd.arg("-W").arg(params.width.unwrap_or(512).to_string());
        cmd.arg("-H").arg(params.height.unwrap_or(512).to_string());
        if let Some(seed) = params.seed {
            cmd.arg("-s").arg(seed.to_string());
        }
        if init_image.is_some() {
            cmd.arg("-i").arg(&init_img_path);
            cmd.arg("--strength").arg(params.strength.unwrap_or(0.75).to_string());
        }
        cmd.arg("-o").arg(&out_path);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        {
            let mut progress = gguf_progress.lock().unwrap();
            progress.active = true;
            progress.step = 0;
            progress.total = total_steps;
            progress.phase = "generating".to_string();
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| BackendError::InferenceError(format!("Failed to run sd.exe: {}", e)))?;

        let stderr = child.stderr.take();
        let stdout = child.stdout.take();
        let mut stderr_buf = Vec::new();

        // sd.exe writes its per-step sampling progress bar to stdout (with
        // `\r`-delimited updates), not stderr; stderr only carries a
        // startup banner and, on failure, error text. Read both streams
        // concurrently: stdout drives live progress, stderr is preserved
        // for its existing error-message role.
        let read_task = if let Some(stderr) = stderr {
            let progress = gguf_progress.clone();
            Some(tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let mut reader = stderr;
                let mut buf = [0u8; 4096];
                let mut line_acc: Vec<u8> = Vec::new();
                let mut all = Vec::new();
                loop {
                    match reader.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(n) => {
                            all.extend_from_slice(&buf[..n]);
                            for &b in &buf[..n] {
                                if b == b'\r' || b == b'\n' {
                                    if !line_acc.is_empty() {
                                        let line = String::from_utf8_lossy(&line_acc).to_string();
                                        if let Some((step, total)) = Self::parse_step_total(&line) {
                                            let mut p = progress.lock().unwrap();
                                            p.step = step;
                                            p.total = total;
                                        }
                                        line_acc.clear();
                                    }
                                } else {
                                    line_acc.push(b);
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
                if !line_acc.is_empty() {
                    let line = String::from_utf8_lossy(&line_acc).to_string();
                    if let Some((step, total)) = Self::parse_step_total(&line) {
                        let mut p = progress.lock().unwrap();
                        p.step = step;
                        p.total = total;
                    }
                }
                all
            }))
        } else {
            None
        };

        let stdout_task = if let Some(stdout) = stdout {
            let progress = gguf_progress.clone();
            Some(tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let mut reader = stdout;
                let mut buf = [0u8; 4096];
                let mut line_acc: Vec<u8> = Vec::new();
                loop {
                    match reader.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(n) => {
                            for &b in &buf[..n] {
                                if b == b'\r' || b == b'\n' {
                                    if !line_acc.is_empty() {
                                        let line = String::from_utf8_lossy(&line_acc).to_string();
                                        if let Some((step, total)) = Self::parse_step_total(&line) {
                                            let mut p = progress.lock().unwrap();
                                            p.step = step;
                                            p.total = total;
                                        }
                                        line_acc.clear();
                                    }
                                } else {
                                    line_acc.push(b);
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
                if !line_acc.is_empty() {
                    let line = String::from_utf8_lossy(&line_acc).to_string();
                    if let Some((step, total)) = Self::parse_step_total(&line) {
                        let mut p = progress.lock().unwrap();
                        p.step = step;
                        p.total = total;
                    }
                }
            }))
        } else {
            None
        };

        // Race the child's exit against a cancellation poll loop, so a
        // caller-set `cancel` flag can abort a running `sd.exe` process.
        let wait_result: Result<std::process::ExitStatus, BackendError> = if let Some(cancel) = cancel {
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
                status = child.wait() => {
                    status.map_err(|e| BackendError::InferenceError(format!("Failed to wait on sd.exe: {}", e)))
                }
                _ = cancel_wait => {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    Err(BackendError::Cancelled)
                }
            }
        } else {
            child
                .wait()
                .await
                .map_err(|e| BackendError::InferenceError(format!("Failed to wait on sd.exe: {}", e)))
        };

        if let Some(task) = read_task {
            if let Ok(bytes) = task.await {
                stderr_buf = bytes;
            }
        }
        if let Some(task) = stdout_task {
            let _ = task.await;
        }

        {
            let mut progress = gguf_progress.lock().unwrap();
            progress.active = false;
        }

        let status = wait_result?;

        if init_image.is_some() {
            let _ = tokio::fs::remove_file(&init_img_path).await;
        }

        if !status.success() {
            let stderr = String::from_utf8_lossy(&stderr_buf).to_string();
            let _ = tokio::fs::remove_file(&out_path).await;
            return Err(BackendError::InferenceError(format!(
                "sd.exe exited with status {:?}: {}",
                status.code(),
                stderr
            )));
        }

        let bytes = tokio::fs::read(&out_path).await.map_err(|e| {
            BackendError::InferenceError(format!("Failed to read sd.exe output PNG: {}", e))
        })?;
        let _ = tokio::fs::remove_file(&out_path).await;
        Ok(bytes)
    }

    /// Generate an image via `sd.exe`. Tries the legacy full-checkpoint `-m`
    /// path first; if that fails with sd.exe's specific "can't sniff
    /// architecture from this file" error, retries once in split-checkpoint
    /// "component mode" (resolved from `component_specs`). Any other failure
    /// is surfaced immediately, unretried.
    async fn generate_gguf(
        sd_binary: &Path,
        model_path: &Path,
        request_id: &str,
        prompt: &str,
        params: &ImageParams,
        gguf_progress: &Arc<Mutex<GgufProgress>>,
        cancel: Option<&Arc<AtomicBool>>,
        init_image: Option<&[u8]>,
    ) -> Result<Vec<u8>, BackendError> {
        let legacy_mode = SdCommandMode::Legacy(model_path.to_path_buf());
        match Self::run_sd_exe(sd_binary, &legacy_mode, request_id, prompt, params, None, None, gguf_progress, cancel, init_image).await {
            Ok(bytes) => Ok(bytes),
            Err(BackendError::InferenceError(msg)) if msg.contains(SD_VERSION_SNIFF_FAILURE) => {
                info!(
                    "'{}' looks like a standalone diffusion-model file (sd.exe couldn't sniff a \
                     full checkpoint architecture from it); retrying in component mode.",
                    model_path.display()
                );
                let (component_mode, default_cfg_scale, default_steps) = Self::resolve_component_mode(model_path)?;
                Self::run_sd_exe(
                    sd_binary,
                    &component_mode,
                    request_id,
                    prompt,
                    params,
                    default_cfg_scale,
                    default_steps,
                    gguf_progress,
                    cancel,
                    init_image,
                )
                .await
            }
            Err(e) => Err(e),
        }
    }

    /// POST the prompt+params to the running `diffusers_server.py` instance
    /// and decode the returned base64 PNG.
    async fn generate_diffusers(
        port: u16,
        prompt: &str,
        params: &ImageParams,
        init_image: Option<&[u8]>,
    ) -> Result<Vec<u8>, BackendError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let mut payload = serde_json::json!({
            "prompt": prompt,
            "negative_prompt": params.negative_prompt,
            "steps": params.steps.unwrap_or(20),
            "guidance_scale": params.cfg_scale.unwrap_or(7.5),
            "width": params.width.unwrap_or(512),
            "height": params.height.unwrap_or(512),
            "seed": params.seed,
        });

        if let Some(bytes) = init_image {
            let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
            payload["init_image_base64"] = serde_json::Value::String(b64);
            payload["strength"] = serde_json::json!(params.strength.unwrap_or(0.75));
        }

        let url = format!("http://127.0.0.1:{}/generate", port);
        let response = client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| BackendError::InferenceError(format!("diffusers_server connection error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let err_text = response.text().await.unwrap_or_default();
            return Err(BackendError::InferenceError(format!(
                "diffusers_server HTTP {} error: {}",
                status, err_text
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| BackendError::InferenceError(format!("diffusers_server response parse error: {}", e)))?;

        let image_b64 = body["image_base64"]
            .as_str()
            .ok_or_else(|| BackendError::InferenceError("diffusers_server response missing image_base64".to_string()))?;

        base64::engine::general_purpose::STANDARD
            .decode(image_b64)
            .map_err(|e| BackendError::InferenceError(format!("Failed to decode image_base64: {}", e)))
    }

    /// GET `/progress` from the running `diffusers_server.py` instance and
    /// map its JSON into a `GenerationProgress`. `job_id`/`modality` are left
    /// for daemon-core to stamp.
    async fn poll_diffusers_progress(port: u16) -> Option<GenerationProgress> {
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
            modality: "image".to_string(),
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

impl Default for SdBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SdBackend {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[async_trait]
impl InferenceBackend for SdBackend {
    fn name(&self) -> &'static str {
        "stable-diffusion"
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
            BackendError::LoadError(format!("Failed to read file metadata for SD VRAM estimate: {}", e))
        })?;

        let file_size = metadata.len();
        // Heuristic: UNet + VAE + TextEncoder VRAM allocation (approx 1.3x weight size)
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
        // Kill any previously running diffusers server and clear the sd.exe
        // binary reference before deciding on the new model's mode.
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.sd_binary = None;

        if !model_path.exists() {
            info!(
                "SD model path '{}' not found on disk; starting backend in simulation mode.",
                model_path.display()
            );
            self.model_path = Some(model_path.to_path_buf());
            self.is_loaded.store(true, Ordering::SeqCst);
            return Ok(());
        }

        let is_gguf = model_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);

        if is_gguf {
            info!("Loading GGUF Stable Diffusion model: {}", model_path.display());
            match Self::find_sd_binary() {
                Some(bin) => {
                    info!("Found stable-diffusion.cpp binary at: {}", bin.display());
                    self.sd_binary = Some(bin);
                }
                None => {
                    warn!(
                        "sd.exe binary not found (set DISPOS_SD_BINARY or place it under \
                         ~/.lmstudio/extensions/backends/); falling back to simulation mode."
                    );
                }
            }
        } else {
            info!("Loading safetensors/HF-repo Stable Diffusion model: {}", model_path.display());
            let port = next_free_sd_port();
            let model_path_owned = model_path.to_path_buf();

            match tokio::task::spawn_blocking(move || Self::spawn_diffusers_server(&model_path_owned, port))
                .await
                .map_err(|e| BackendError::LoadError(format!("spawn_blocking join error: {}", e)))?
            {
                Ok(child) => {
                    info!("diffusers_server.py ready on port {} (PID: {})", port, child.id());
                    self.process = Some(child);
                    self.port = port;
                }
                Err(e) => {
                    warn!(
                        "Could not start the diffusers server for '{}': {}. Falling back to simulation mode.",
                        model_path.display(),
                        e
                    );
                }
            }
        }

        self.model_path = Some(model_path.to_path_buf());
        self.is_loaded.store(true, Ordering::SeqCst);

        info!("Successfully loaded model in Stable Diffusion backend");
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<(), BackendError> {
        if self.is_loaded.load(Ordering::SeqCst) {
            info!("Unloading Stable Diffusion model from memory");
            if let Some(mut child) = self.process.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
            self.sd_binary = None;
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
        let params = request.image_params.clone().unwrap_or_default();

        info!("Executing image generation prompt: '{}'", request.prompt);

        let png_bytes: Vec<u8> = if let Some(sd_binary) = &self.sd_binary {
            let model_path = self.model_path.as_deref().ok_or(BackendError::ModelNotLoaded)?;
            Self::generate_gguf(
                sd_binary,
                model_path,
                &request.request_id,
                &request.prompt,
                &params,
                &self.gguf_progress,
                request.cancel.as_ref(),
                request.image_input.as_deref(),
            )
            .await?
        } else if self.process.is_some() {
            Self::generate_diffusers(self.port, &request.prompt, &params, request.image_input.as_deref()).await?
        } else {
            // Simulation mode: no sd.exe binary and no diffusers server running
            // (e.g. missing model file or missing runtime dependencies).
            vec![
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
                0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
                0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
                0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
                0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
                0x42, 0x60, 0x82,
            ]
        };

        let duration = start_time.elapsed().as_millis() as u64;

        Ok(InferenceResponse {
            request_id: request.request_id,
            output_text: "Image generation completed successfully".to_string(),
            output_data: Some(png_bytes),
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
            delta_text: "Generating image steps...".to_string(),
            delta_data: None,
            is_final: true,
            delta_tool_call: None,
        })];

        Ok(Box::pin(stream::iter(chunks)))
    }

    async fn poll_progress(&self) -> Option<GenerationProgress> {
        if self.sd_binary.is_some() {
            let progress = self.gguf_progress.lock().unwrap();
            if !progress.active {
                return None;
            }
            let percent = if progress.total > 0 {
                progress.step as f32 / progress.total as f32 * 100.0
            } else {
                -1.0
            };
            let updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            return Some(GenerationProgress {
                job_id: String::new(),
                modality: String::new(),
                phase: progress.phase.clone(),
                step: progress.step,
                total: progress.total,
                percent,
                status: "running".to_string(),
                message: None,
                media_handle: None,
                media_type: None,
                updated_at,
            });
        }

        if self.process.is_none() {
            // No diffusers_server.py running (simulation mode) — no progress
            // source available.
            return None;
        }
        Self::poll_diffusers_progress(self.port).await
    }
}
