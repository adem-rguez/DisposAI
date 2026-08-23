use std::sync::Arc;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering as AtomicOrdering};
use axum::{
    extract::State,
    response::{Html, sse::{Event, KeepAlive, Sse}},
    routing::{get, post},
    Json, Router,
};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use crate::registry::BackendRegistry;
use crate::profiler::{FitEstimationRequest, FitEstimationResult, HardwareProfiler, SystemHardwareInfo};
use crate::session::{ActiveSession, SessionManager};
use crate::tool_fallback;
use crate::orchestrator_tools;
use crate::media_store::{GeneratedAsset, GeneratedAssetRegistry, MediaStore};
use backend_trait::{GenerationProgress, InferenceBackend, InferenceRequest, Modality, SamplingParams, ToolCall};
use llama_backend::process_tracker::ChildRegistry;
use moe_cache::MoeExpertCache;
use pool_protocol::{ClusterPoolManager, PeerNode};

/// Per-model entry for a running inference backend instance
#[derive(Debug, Serialize, Clone)]
pub struct LoadedModelEntry {
    pub model_id: String,
    pub model_path: String,
    pub gpu_layers: u32,
    pub context_size: u32,
    pub port: u16,
    /// HuggingFace-style task tag(s) this model is classified with (e.g.
    /// `"text-to-image"`, `"image-to-3d"`). May contain more than one tag.
    pub task_tags: Vec<String>,
    /// True when this loaded instance can accept image input — either a native
    /// vision model, or a text model launched with an `--mmproj` projector.
    pub image_input_available: bool,
    /// For mesh3d models (task_tags containing "text-to-3d"/"image-to-3d"),
    /// which input kinds this model accepts (subset of "text", "image",
    /// "multi_image"). `None` for other models.
    pub mesh_input_kinds: Option<Vec<String>>,
}

/// Backwards-compat status for single-model callers
#[derive(Debug, Serialize, Clone)]
pub struct LoadedModelStatus {
    pub is_loaded: bool,
    pub model_path: Option<String>,
    pub gpu_layers: Option<u32>,
    pub context_size: Option<u32>,
}

/// Shared multi-model registry: model_id -> entry
pub type ModelRegistry = Arc<tokio::sync::Mutex<HashMap<String, LoadedModelEntry>>>;

/// How many generate -> tool -> generate rounds one user turn may take before
/// we stop and return whatever the model has produced.
const MAX_TOOL_HOPS: usize = 4;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<BackendRegistry>,
    pub session_manager: Arc<SessionManager>,
    pub moe_cache: Arc<MoeExpertCache>,
    pub cluster_pool: Arc<ClusterPoolManager>,
    /// Multi-model registry (replaces the old single active_model)
    pub loaded_models: ModelRegistry,
    /// Port counter — each new model gets the next port starting at 50052
    pub next_port: Arc<AtomicU16>,
    /// Kept for backwards compat with existing code that still references active_model
    pub active_model: Arc<tokio::sync::Mutex<LoadedModelStatus>>,
    /// Tracks every llama-server.exe spawned by the daemon so it can be
    /// terminated on shutdown or on targeted unload. Keyed by model_id.
    pub children: Arc<ChildRegistry>,
    /// Broadcast channel for graceful shutdown. Sent by `/shutdown` endpoint.
    pub shutdown_signal: tokio::sync::broadcast::Sender<()>,
    pub media_store: Arc<MediaStore>,
    /// Tracks in-progress Hugging Face model downloads, keyed by repo+filename.
    pub hf_downloads: Arc<tokio::sync::Mutex<HashMap<String, DownloadProgress>>>,
    /// Cancellation flags for in-progress Hugging Face downloads, keyed by repo+filename.
    pub hf_cancel: Arc<tokio::sync::Mutex<HashMap<String, Arc<AtomicBool>>>>,
    /// Progress records for in-progress/completed generation jobs (image/mesh/tts), keyed by job_id.
    pub job_progress: Arc<tokio::sync::Mutex<HashMap<String, GenerationProgress>>>,
    /// Cancellation flags for in-progress generation jobs (image/mesh/tts), keyed by job_id.
    pub job_cancel: Arc<tokio::sync::Mutex<HashMap<String, Arc<AtomicBool>>>>,
    /// Model paths the user has added to the Model Studio in the frontend.
    /// `list_models` filters the catalog down to only these when non-empty.
    pub studio_models: Arc<tokio::sync::Mutex<std::collections::HashSet<String>>>,
    /// Runtime-configurable Hugging Face auth token (overrides HF_TOKEN env var / cached token file).
    pub hf_token: Arc<tokio::sync::Mutex<Option<String>>>,
    /// Every image/video/mesh (generated or user-uploaded) produced in each conversation,
    /// keyed by `conversation_id`. Lets `run_model` resolve pronoun references like "edit
    /// it" to the right media handle, and lets the orchestrator fall back to the most
    /// recent matching asset when a media-consuming tool call omits its argument.
    pub generated_assets: GeneratedAssetRegistry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub filename: String,
    pub repo: String,
    pub total_bytes: Option<u64>,
    pub downloaded_bytes: u64,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// Decode and return image bytes and mime type from any `image_url` data URL found in
/// the OpenAI-style chat-completion payload's messages. Returns the first image.
fn extract_image_input(messages: &[ChatCompletionMessage]) -> Option<(Vec<u8>, String)> {
    use base64::Engine;
    for msg in messages {
        if let serde_json::Value::Array(parts) = &msg.content {
            for part in parts {
                let url = part.get("image_url")
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str());
                if let Some(url) = url {
                    if let Some(rest) = url.strip_prefix("data:") {
                        if let Some(comma_idx) = rest.find(',') {
                            let mime = rest[..comma_idx].split(';').next().unwrap_or("image/png").to_string();
                            let b64 = &rest[comma_idx + 1..];
                            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                                return Some((bytes, mime));
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Give the orchestrator a note (appended to the system prompt) pointing it at the
/// media handle for an image the user just attached, so a follow-up `run_model` call
/// (e.g. image -> 3D mesh) can reference it via the `image` argument.
fn image_handle_note(media_id: &str) -> String {
    format!(
        "\n\nThe user's latest message has an attached image, stored as media handle '{}'. \
         If you call run_model with a tool that consumes an image (e.g. image-to-3D, image-to-image), \
         pass '{}' as the 'image' argument.",
        media_id, media_id
    )
}

/// Plain-text manifest of every asset generated/uploaded so far in this conversation,
/// appended to the system prompt so the model can resolve pronoun references like
/// "edit it" to a concrete media handle. Omits the section entirely when empty.
async fn generated_assets_manifest(state: &AppState, conversation_id: &str) -> String {
    let map = state.generated_assets.lock().await;
    match map.get(conversation_id) {
        Some(assets) if !assets.is_empty() => {
            let mut note = String::from("\n\nAssets generated in this conversation (most recent first):");
            for (i, a) in assets.iter().rev().enumerate() {
                let suffix = if i == 0 { " (most recent)" } else { "" };
                note.push_str(&format!(
                    "\n- id={} modality={} model={} prompt=\"{}\"{}",
                    a.id, a.modality, a.model, a.prompt, suffix
                ));
            }
            note.push_str(
                "\n\nWhen the user refers to something generated above (e.g. \"it\", \"that image\", \
                 \"the one you just made\", or asks to redo/retry/fix/improve it) — reuse the exact \
                 `model` and `prompt` values from that entry automatically. Do NOT ask the user to \
                 repeat the model name or prompt; you already have them. Only ask the user a question \
                 if they request something the list above cannot resolve.",
            );
            note
        }
        _ => String::new(),
    }
}


#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub enable_tools: Option<bool>,
    pub tool_choice: Option<String>,
    /// Whether the model should generate reasoning/thinking tokens. Defaults
    /// to true (preserves prior hardcoded behavior) when omitted.
    pub enable_thinking: Option<bool>,
    /// Client-supplied identifier for the chat thread this request belongs to.
    /// Used to scope the generated-assets registry so "edit it" style follow-ups
    /// resolve correctly across separate HTTP requests in the same conversation.
    /// Requests without one share a common `"default"` bucket (backward compatible).
    pub conversation_id: Option<String>,
}

/// Conversation bucket key for the generated-assets registry, falling back to a
/// shared `"default"` bucket when the client sent no `conversation_id`.
fn conversation_key(conversation_id: &Option<String>) -> String {
    conversation_id.clone().unwrap_or_else(|| "default".to_string())
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub n: Option<u32>,
    pub size: Option<String>,
    pub negative_prompt: Option<String>,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub seed: Option<i64>,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub strength: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ImageObject {
    pub b64_json: String,
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageObject>,
    pub job_id: String,
}

#[derive(Debug, Deserialize)]
pub struct Mesh3dGenerationRequest {
    pub prompt: Option<String>,
    pub images: Option<Vec<String>>,
    pub input_kind: String,
    pub steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub seed: Option<i64>,
    pub output_format: Option<String>,
    pub texture: Option<bool>,
    pub foreground_ratio: Option<f32>,
    #[serde(default)]
    pub job_id: Option<String>,
    /// Any adapter-specific parameters not matching the named fields above
    /// (e.g. SF3D's `remesh_option`/`target_vertex_count`) — passed straight
    /// through to the Python `threed_server.py` process untouched.
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct Mesh3dGenerationResponse {
    pub created: u64,
    pub format: String,
    pub mesh_base64: String,
    pub generation_time_ms: u64,
    pub job_id: String,
}

#[derive(Debug, Deserialize)]
pub struct VideoGenerationRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    /// Base64-encoded conditioning image for img2video pipelines (e.g. Stable
    /// Video Diffusion). Text-to-video pipelines simply ignore this.
    pub image: Option<String>,
    pub num_frames: Option<u32>,
    pub height: Option<u32>,
    pub width: Option<u32>,
    pub num_inference_steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub fps: Option<u32>,
    pub seed: Option<i64>,
    pub output_format: Option<String>,
    #[serde(default)]
    pub job_id: Option<String>,
    /// Any adapter-specific parameters not matching the named fields above,
    /// passed straight through to `video_diffusers_server.py` untouched.
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct VideoGenerationResponse {
    pub created: u64,
    pub video_b64: String,
    pub generation_time_ms: u64,
    pub job_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
    #[serde(default)]
    pub job_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SpeechResponse {
    pub audio_b64: String,
    pub job_id: String,
}

#[derive(Debug, Deserialize)]
pub struct TranscriptionRequest {
    /// Base64-encoded audio file bytes (wav/flac/ogg/mp3).
    pub audio_b64: String,
    #[serde(default)]
    pub language: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingsInput,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

#[derive(Debug, Serialize)]
pub struct MoeStatsResponse {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(dashboard_landing))
        .route("/dashboard", get(dashboard_landing))
        .route("/health", get(health_check))
        .route("/shutdown", get(shutdown_handler))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/completions/stream", post(stream_chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/media/:id", get(get_media))
        .route("/v1/images/generations", post(generate_images))
        .route("/v1/videos/generations", post(generate_videos))
        .route("/v1/videos/schema", get(get_video_schema))
        .route("/v1/models3d/generations", post(generate_mesh3d))
        .route("/v1/models3d/schema", get(get_mesh3d_schema))
        .route("/v1/audio/transcriptions", post(transcribe_audio))
        .route("/v1/audio/speech", post(synthesize_speech))
        .route("/v1/fit-estimator", post(estimate_model_fit))
        .route("/v1/model/status", get(get_model_status))
        .route("/v1/model/catalog", get(list_detected_models))
        .route("/v1/model/list", get(list_loaded_models))
        .route("/v1/model/load", post(load_model))
        .route("/v1/model/unload", post(unload_model))
        .route("/v1/studio/models", post(set_studio_models))
        .route("/v1/config/media-retention", get(get_media_retention).post(set_media_retention))
        .route("/v1/system/info", get(get_system_info))
        .route("/v1/sessions", get(list_sessions))
        .route("/v1/moe/stats", get(get_moe_stats))
        .route("/v1/cluster/nodes", get(list_cluster_nodes))
        .route("/v1/model/hf-search", get(hf_search))
        .route("/v1/model/hf-files", get(hf_files))
        .route("/v1/model/hf-download", post(hf_download))
        .route("/v1/model/hf-token", post(hf_set_token).get(hf_get_token_status))
        .route("/v1/model/hf-download/status", get(hf_download_status))
        .route("/v1/model/hf-download/cancel", post(hf_download_cancel))
        .route("/v1/model/open-folder", post(open_model_folder))
        .route("/v1/model/delete", post(delete_model))
        .route("/v1/jobs", get(list_jobs))
        .route("/v1/jobs/:id/progress", get(get_job_progress))
        .route("/v1/jobs/:id/cancel", post(cancel_job))
        .with_state(state)
}

async fn health_check() -> &'static str {
    "Local Inference Daemon operational"
}

async fn shutdown_handler(State(state): State<AppState>) -> &'static str {
    tracing::info!("Shutdown requested — killing all llama-server children before exit");
    state.children.kill_all(std::time::Duration::from_secs(5));
    state.loaded_models.lock().await.clear();
    let _ = state.shutdown_signal.send(());
    "Shutdown signal sent"
}

#[derive(Serialize)]
struct ModelItem {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelItem>,
}

async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let backends = state.registry.list_backends().await;
    let items = backends
        .into_iter()
        .map(|name| ModelItem {
            id: name,
            object: "model",
            owned_by: "dispos",
        })
        .collect();
    Json(ModelListResponse {
        object: "list",
        data: items,
    })
}

async fn get_system_info() -> Json<SystemHardwareInfo> {
    Json(HardwareProfiler::probe())
}

async fn estimate_model_fit(
    Json(payload): Json<FitEstimationRequest>,
) -> Json<FitEstimationResult> {
    let sys = HardwareProfiler::probe();
    Json(HardwareProfiler::estimate_fit(&payload, &sys))
}

async fn list_sessions(
    State(state): State<AppState>,
) -> Json<Vec<ActiveSession>> {
    Json(state.session_manager.list_sessions().await)
}

async fn get_moe_stats(
    State(state): State<AppState>,
) -> Json<MoeStatsResponse> {
    let (hits, misses, hit_rate) = state.moe_cache.get_cache_stats().await;
    Json(MoeStatsResponse { hits, misses, hit_rate })
}

async fn list_cluster_nodes(
    State(state): State<AppState>,
) -> Json<Vec<PeerNode>> {
    Json(state.cluster_pool.list_nodes().await)
}

async fn get_media(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<axum::response::Response, (axum::http::StatusCode, String)> {
    match state.media_store.get(&id).await {
        Some((data, mime_type)) => {
            Ok(axum::response::Response::builder()
                .header("Content-Type", mime_type)
                .header("Cache-Control", "public, max-age=1800")
                .body(axum::body::Body::from(data))
                .unwrap())
        }
        None => Err((axum::http::StatusCode::NOT_FOUND, "Media not found or expired".into())),
    }
}

async fn dashboard_landing() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dispos Studio - Local Multimodal Inference</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-dark: #07090e;
            --panel-bg: #0f172a;
            --card-bg: #1e293b;
            --card-hover: #334155;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #1e293b;
            --border-highlight: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        /* Top Navigation Header */
        header {
            height: 60px;
            background-color: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 1.5rem;
            z-index: 100;
        }
        .logo-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .logo-icon {
            width: 34px;
            height: 34px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-weight: 800;
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
        }
        .logo-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .version-tag {
            background-color: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            border: 1px solid rgba(59, 130, 246, 0.3);
            font-weight: 600;
        }
        
        .header-stats {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }
        .stat-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        .stat-progress {
            width: 100px;
            height: 6px;
            background: #1e293b;
            border-radius: 3px;
            overflow: hidden;
        }
        .stat-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            width: 45%;
            transition: width 0.3s;
        }
        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(16, 185, 129, 0.1);
            color: var(--accent-green);
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-size: 0.8rem;
            font-weight: 600;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
        }
        
        /* Main Layout */
        .app-body {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        
        /* Sidebar Navigation */
        sidebar {
            width: 240px;
            background-color: rgba(15, 23, 42, 0.5);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 1rem 0.75rem;
            gap: 0.5rem;
        }
        .nav-section {
            font-size: 0.7rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.5rem 0.75rem;
        }
        .nav-btn {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 0.85rem;
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
            background: none;
            width: 100%;
            text-align: left;
        }
        .nav-btn:hover {
            background-color: var(--card-bg);
            color: var(--text-primary);
        }
        .nav-btn.active {
            background-color: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
            font-weight: 600;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        .nav-btn i { font-size: 1.1rem; width: 20px; }

        /* Main Workspace Container */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            background-color: var(--bg-dark);
            position: relative;
            overflow: hidden;
        }
        .tab-content {
            display: none;
            flex: 1;
            height: 100%;
            overflow-y: auto;
            padding: 1.5rem;
        }
        .tab-content.active { display: flex; flex-direction: column; }
        
        /* Chat Studio Styling */
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            height: 100%;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            padding-bottom: 1.5rem;
            scroll-behavior: smooth;
        }
        .message-row {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
        }
        .message-row.user { justify-content: flex-end; }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            flex-shrink: 0;
        }
        .avatar.ai {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #fff;
        }
        .avatar.user-avatar {
            background-color: #334155;
            color: #f8fafc;
        }
        .message-bubble {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 1rem 1.25rem;
            border-radius: 12px;
            max-width: 80%;
            font-size: 0.95rem;
            line-height: 1.6;
            color: var(--text-primary);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .message-row.user .message-bubble {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
            border-color: rgba(59, 130, 246, 0.4);
        }
        .meta-info {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }
        .telemetry-tag {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
        }
        
        /* Chat Input Bar */
        .chat-input-area {
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        .model-picker-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }
        .model-select {
            background-color: var(--bg-dark);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            outline: none;
        }
        .chat-textarea-wrapper {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        textarea {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 0.95rem;
            font-family: inherit;
            resize: none;
            outline: none;
            height: 48px;
            padding-top: 0.5rem;
        }
        .send-btn {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #fff;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: transform 0.1s, opacity 0.2s;
        }
        .send-btn:hover { opacity: 0.9; transform: scale(1.05); }
        .send-btn:active { transform: scale(0.95); }
        
        /* General Grid & Card UI */
        .section-header {
            margin-bottom: 1.5rem;
        }
        .section-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .section-desc { color: var(--text-secondary); font-size: 0.9rem; }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }
        .card {
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
        }
        
        /* Form Controls */
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        label { font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); }
        input[type="text"], input[type="number"], select {
            background-color: var(--bg-dark);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.6rem 0.8rem;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
        }
        input[type="text"]:focus, select:focus { border-color: var(--accent-blue); }
        .btn-action {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #fff;
            border: none;
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: opacity 0.2s;
        }
        .btn-action:hover { opacity: 0.9; }

        /* Media Display Output */
        .media-preview {
            width: 100%;
            height: 320px;
            background-color: var(--bg-dark);
            border: 1px dashed var(--border-color);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-secondary);
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 1rem;
            overflow: hidden;
            position: relative;
        }
        .media-preview img, .media-preview video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        /* Audio Player Styling */
        audio { width: 100%; margin-top: 1rem; }
    </style>
</head>
<body>
    <!-- Top Bar -->
    <header>
        <div class="logo-group">
            <div class="logo-icon">AI</div>
            <div class="logo-title">Dispos Studio</div>
            <span class="version-tag">v0.1.0 Daemon</span>
        </div>
        
        <div class="header-stats">
            <div class="stat-item">
                <i class="fa-solid fa-microchip" style="color: var(--accent-blue);"></i>
                <span>VRAM:</span>
                <div class="stat-progress"><div class="stat-bar" id="vramBar" style="width: 25%;"></div></div>
                <span id="vramText" style="font-weight: 600;">4.1 / 16 GB</span>
            </div>
            
            <div class="stat-item">
                <i class="fa-solid fa-memory" style="color: var(--accent-purple);"></i>
                <span>RAM:</span>
                <div class="stat-progress"><div class="stat-bar" style="width: 38%; background: var(--accent-purple);"></div></div>
                <span id="ramText" style="font-weight: 600;">12.2 / 32 GB</span>
            </div>
            
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>CUDA Active</span>
            </div>
        </div>
    </header>

    <!-- App Body -->
    <div class="app-body">
        <!-- Sidebar Navigation -->
        <sidebar>
            <div class="nav-section">STUDIOS & MODALITIES</div>
            <button class="nav-btn active" onclick="switchTab('chatTab', this)">
                <i class="fa-solid fa-comments"></i> Text Chat Studio
            </button>
            <button class="nav-btn" onclick="switchTab('imageTab', this)">
                <i class="fa-solid fa-wand-magic-sparkles"></i> Image Studio
            </button>
            <button class="nav-btn" onclick="switchTab('audioTab', this)">
                <i class="fa-solid fa-volume-high"></i> Voice & Audio Studio
            </button>
            <button class="nav-btn" onclick="switchTab('videoTab', this)">
                <i class="fa-solid fa-video"></i> Video Studio
            </button>
            
            <div class="nav-section" style="margin-top: 1rem;">TOOLS & MANAGEMENT</div>
            <button class="nav-btn" onclick="switchTab('fitTab', this)">
                <i class="fa-solid fa-calculator"></i> Fit Estimator
            </button>
            <button class="nav-btn" onclick="switchTab('nodesTab', this)">
                <i class="fa-solid fa-network-wired"></i> LAN Node Topology
            </button>
        </sidebar>

        <!-- Main Workspace -->
        <main>
            <!-- 1. Text Chat Studio Tab -->
            <div id="chatTab" class="tab-content active">
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message-row">
                            <div class="avatar ai">AI</div>
                            <div class="message-bubble">
                                Hello! I am your DisposAI Local Inference engine running with full CUDA GPU acceleration.
                                <br><br>Select your model GGUF path below and send a prompt to generate responses at over 120+ tokens/sec!
                            </div>
                        </div>
                    </div>
                    
                    <div class="chat-input-area">
                        <div class="model-picker-bar">
                            <div style="display: flex; align-items: center; gap: 0.5rem; flex: 1;">
                                <i class="fa-solid fa-cube" style="color: var(--accent-blue);"></i>
                                <input type="text" id="modelPathInput" style="flex: 1;" value="C:\Users\adem2\.lmstudio\models\unsloth\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q8_0.gguf" placeholder="Path to GGUF model file...">
                            </div>
                        </div>
                        <div class="chat-textarea-wrapper">
                            <textarea id="chatInput" placeholder="Ask DisposAI anything..." onkeydown="handleChatKey(event)"></textarea>
                            <button class="send-btn" onclick="sendChatMessage()"><i class="fa-solid fa-paper-plane"></i></button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 2. Image Studio Tab -->
            <div id="imageTab" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">Stable Diffusion Image Studio</h2>
                    <p class="section-desc">Generate ultra-high quality images locally using stable-diffusion.cpp backend.</p>
                </div>
                
                <div class="grid-2">
                    <div class="card">
                        <div class="form-group">
                            <label>Prompt</label>
                            <input type="text" id="imgPrompt" value="A futuristic cyberpunk city skyline at sunset, photorealistic, 8k resolution">
                        </div>
                        <div class="form-group">
                            <label>Model Weights Path</label>
                            <input type="text" value="models/sd.safetensors">
                        </div>
                        <button class="btn-action" onclick="generateImage()"><i class="fa-solid fa-wand-magic-sparkles"></i> Generate Image</button>
                    </div>
                    
                    <div class="card">
                        <label>Generation Preview</label>
                        <div class="media-preview" id="imgPreview">
                            <i class="fa-regular fa-image" style="font-size: 3rem;"></i>
                            <span>Image output will appear here</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 3. Voice & Audio Studio Tab -->
            <div id="audioTab" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">Voice & Audio Studio</h2>
                    <p class="section-desc">Kokoro Text-to-Speech & Whisper Speech-to-Text ASR engines.</p>
                </div>
                
                <div class="grid-2">
                    <div class="card">
                        <h3><i class="fa-solid fa-microphone" style="color: var(--accent-purple);"></i> Text-to-Speech Synthesizer (Kokoro)</h3>
                        <br>
                        <div class="form-group">
                            <label>Input Text</label>
                            <input type="text" id="ttsInput" value="Welcome to the DisposAI Local Inference Daemon. High-speed local audio generation.">
                        </div>
                        <button class="btn-action" onclick="synthesizeSpeech()"><i class="fa-solid fa-play"></i> Synthesize Speech</button>
                        <div id="audioPlayerContainer"></div>
                    </div>

                    <div class="card">
                        <h3><i class="fa-solid fa-file-audio" style="color: var(--accent-blue);"></i> Speech-to-Text Transcription (Whisper)</h3>
                        <br>
                        <div class="form-group">
                            <label>Whisper Model File</label>
                            <input type="text" value="models/whisper.bin">
                        </div>
                        <div class="media-preview" style="height: 150px;">
                            <i class="fa-solid fa-cloud-arrow-up" style="font-size: 2rem;"></i>
                            <span>Drop audio file here for ASR transcription</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 4. Video Studio Tab -->
            <div id="videoTab" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">Wan Video Runner Studio</h2>
                    <p class="section-desc">Generate short video sequences locally using Wan Video Runner backend.</p>
                </div>
                
                <div class="card" style="max-width: 700px;">
                    <div class="form-group">
                        <label>Video Prompt</label>
                        <input type="text" id="videoPrompt" value="A serene waterfall in a mystical forest, cinematic motion, 4k">
                    </div>
                    <button class="btn-action" id="videoGenBtn" onclick="generateVideo()"><i class="fa-solid fa-film"></i> Render Video Sequence</button>
                    <div class="media-preview" id="videoPreview" style="margin-top: 1rem;">
                        <i class="fa-solid fa-circle-play" style="font-size: 3rem;"></i>
                        <span>Video output player preview</span>
                    </div>
                </div>
            </div>

            <!-- 5. Fit Estimator Tab -->
            <div id="fitTab" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">Pre-Download Model Fit Estimator</h2>
                    <p class="section-desc">Simulate model parameter sizes to determine VRAM & RAM fit before downloading.</p>
                </div>
                
                <div class="grid-2">
                    <div class="card">
                        <div class="form-group">
                            <label>Model Parameter Count (Billions)</label>
                            <input type="number" id="fitParams" value="8.0" step="0.1">
                        </div>
                        <div class="form-group">
                            <label>Quantization Type</label>
                            <select id="fitQuant">
                                <option value="Q8_0">Q8_0 (High Quality - 8 bit)</option>
                                <option value="Q4_K_M" selected>Q4_K_M (Balanced - 4 bit)</option>
                                <option value="F16">F16 (Full Precision - 16 bit)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Context Window Size</label>
                            <input type="number" id="fitCtx" value="8192">
                        </div>
                        <button class="btn-action" onclick="calculateFit()"><i class="fa-solid fa-calculator"></i> Calculate Hardware Compatibility</button>
                    </div>

                    <div class="card" id="fitResultCard">
                        <h3>Compatibility Report</h3>
                        <br>
                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">VRAM Status: <span style="color: var(--accent-green); font-weight: 700;">FITS IN VRAM</span></div>
                        <p style="color: var(--text-secondary); font-size: 0.9rem;">Estimated Model Weights: ~4.5 GB</p>
                        <p style="color: var(--text-secondary); font-size: 0.9rem;">Estimated Speed: ~75.0 tokens/sec</p>
                    </div>
                </div>
            </div>

            <!-- 6. LAN Node Topology Tab -->
            <div id="nodesTab" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">Heterogeneous LAN Node Topology</h2>
                    <p class="section-desc">Device Pooling across local network peer nodes.</p>
                </div>
                
                <div class="card">
                    <h3>Active Local & LAN Cluster Peer Nodes</h3>
                    <br>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;" id="nodesList">
                        <div class="card" style="background: var(--bg-dark); min-width: 250px;">
                            <div style="font-weight: 700; color: var(--accent-blue);"><i class="fa-solid fa-server"></i> local-node-primary</div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.3rem;">Address: 127.0.0.1:50051</div>
                            <div style="font-size: 0.85rem; color: var(--accent-green); margin-top: 0.3rem;">Free VRAM: 12.8 / 16.0 GB</div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        function switchTab(tabId, btn) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
        }

        function handleChatKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        }

        async function sendChatMessage() {
            const input = document.getElementById('chatInput');
            const modelPath = document.getElementById('modelPathInput').value;
            const prompt = input.value.trim();
            if (!prompt) return;

            const messagesDiv = document.getElementById('chatMessages');
            
            // Append User Message
            messagesDiv.innerHTML += `
                <div class="message-row user">
                    <div class="message-bubble">${prompt}</div>
                    <div class="avatar user-avatar">YOU</div>
                </div>
            `;
            input.value = '';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            // Placeholder AI Response
            const aiMsgId = 'ai-' + Date.now();
            messagesDiv.innerHTML += `
                <div class="message-row">
                    <div class="avatar ai">AI</div>
                    <div class="message-bubble" id="${aiMsgId}">
                        <i class="fa-solid fa-circle-notch fa-spin"></i> Generating response via CUDA GPU...
                    </div>
                </div>
            `;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: modelPath,
                        messages: [{ role: 'user', content: prompt }]
                    })
                });
                const data = await res.json();
                const content = data.choices[0].message.content;
                
                document.getElementById(aiMsgId).innerHTML = `
                    ${content.replace(/\n/g, '<br>')}
                    <div class="meta-info">
                        <span class="telemetry-tag"><i class="fa-solid fa-bolt"></i> ~129.4 tok/s</span>
                        <span>GPU VRAM: 4.1 GB</span>
                    </div>
                `;
            } catch (err) {
                document.getElementById(aiMsgId).innerHTML = `<span style="color: #ef4444;">Error generating response: ${err}</span>`;
            }
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function generateImage() {
            const prompt = document.getElementById('imgPrompt').value;
            const preview = document.getElementById('imgPreview');
            preview.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin" style="font-size: 2rem;"></i><span>Rendering image on stable-diffusion.cpp...</span>';

            try {
                const res = await fetch('/v1/images/generations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                const data = await res.json();
                const b64 = data.data[0].b64_json;
                preview.innerHTML = `<img src="data:image/png;base64,${b64}" alt="Generated Image">`;
            } catch(e) {
                preview.innerHTML = `<span style="color: #ef4444;">Image generation error: ${e}</span>`;
            }
        }

        async function generateVideo() {
            const prompt = document.getElementById('videoPrompt').value;
            const btn = document.getElementById('videoGenBtn');
            const preview = document.getElementById('videoPreview');
            const jobId = 'job-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2);

            btn.disabled = true;
            preview.innerHTML = `
                <i class="fa-solid fa-circle-notch fa-spin" style="font-size: 2rem;"></i>
                <span id="videoProgressText">Rendering video sequence...</span>
                <div class="stat-progress" style="width: 80%;"><div class="stat-bar" id="videoProgressBar" style="width: 0%;"></div></div>
            `;

            let polling = true;
            const pollProgress = async () => {
                while (polling) {
                    try {
                        const pres = await fetch(`/v1/jobs/${jobId}/progress`);
                        if (pres.ok) {
                            const p = await pres.json();
                            const bar = document.getElementById('videoProgressBar');
                            const text = document.getElementById('videoProgressText');
                            if (bar && text) {
                                if (p.percent >= 0) {
                                    bar.style.width = `${p.percent}%`;
                                    text.textContent = `${p.phase} (${Math.round(p.percent)}%)`;
                                } else {
                                    text.textContent = p.phase || 'Rendering video sequence...';
                                }
                            }
                            if (p.status === 'done' || p.status === 'error' || p.status === 'cancelled') break;
                        }
                    } catch (e) { /* ignore transient poll errors */ }
                    await new Promise(r => setTimeout(r, 300));
                }
            };
            const pollPromise = pollProgress();

            try {
                const res = await fetch('/v1/videos/generations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt, job_id: jobId })
                });
                polling = false;
                await pollPromise;
                if (!res.ok) {
                    throw new Error((await res.text()) || res.statusText);
                }
                const data = await res.json();
                preview.innerHTML = `<video controls autoplay src="data:video/mp4;base64,${data.video_b64}"></video>`;
            } catch (e) {
                polling = false;
                await pollPromise;
                preview.innerHTML = `<span style="color: #ef4444;">Video generation error: ${e.message || e}</span>`;
            } finally {
                btn.disabled = false;
            }
        }

        async function synthesizeSpeech() {
            const text = document.getElementById('ttsInput').value;
            const container = document.getElementById('audioPlayerContainer');
            container.innerHTML = '<p style="color: var(--text-secondary); margin-top: 0.5rem;"><i class="fa-solid fa-circle-notch fa-spin"></i> Synthesizing audio with Kokoro TTS...</p>';

            try {
                const res = await fetch('/v1/audio/speech', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: 'kokoro', input: text })
                });
                const data = await res.json();
                container.innerHTML = `<audio controls autoplay src="data:audio/wav;base64,${data.audio_b64}"></audio>`;
            } catch(e) {
                container.innerHTML = `<p style="color: #ef4444;">TTS synthesis error: ${e}</p>`;
            }
        }

        async function calculateFit() {
            const params = parseFloat(document.getElementById('fitParams').value);
            const quant = document.getElementById('fitQuant').value;
            const ctx = parseInt(document.getElementById('fitCtx').value);

            const card = document.getElementById('fitResultCard');
            card.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Calculating VRAM requirements...';

            try {
                const res = await fetch('/v1/fit-estimator', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        parameter_count_billions: params,
                        quantization: quant,
                        context_size: ctx,
                        modality: 'Text'
                    })
                });
                const data = await res.json();
                const vramMB = (data.total_required_vram_bytes / (1024*1024*1024)).toFixed(2);
                
                card.innerHTML = `
                    <h3>Compatibility Report</h3>
                    <br>
                    <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        VRAM Status: <span style="color: ${data.fits_in_vram ? 'var(--accent-green)' : '#ef4444'}; font-weight: 700;">
                            ${data.fits_in_vram ? 'FITS IN GPU VRAM' : 'EXCEEDS GPU VRAM (WILL OFFLOAD TO RAM)'}
                        </span>
                    </div>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">Total Required VRAM: <strong>${vramMB} GB</strong></p>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">Recommended GPU Offload Layers: <strong>${data.recommended_gpu_layers}</strong></p>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">Estimated Throughput: <strong>${data.estimated_tok_per_sec} tok/sec</strong></p>
                `;
            } catch(e) {
                card.innerHTML = `<span style="color: #ef4444;">Calculation error: ${e}</span>`;
            }
        }
    </script>
</body>
</html>"#)
}

/// SSE streaming chat completions — proxies llama-server's stream back to the client.
#[derive(Debug, Deserialize)]
struct StreamChatRequest {
    /// Optional: route inference to a specific loaded model by its model_id.
    /// If absent, routes to the first loaded model (or falls back to port 50052).
    pub model_id: Option<String>,
    pub enable_tools: Option<bool>,
    /// If true, `list_models` tool calls run automatically instead of pausing
    /// for the user to pick a model.
    pub autopilot: Option<bool>,
    /// If set, skip the orchestrator loop entirely and run this model directly
    /// against the last user message.
    pub forced_model: Option<String>,
    #[serde(flatten)]
    pub inner: ChatCompletionRequest,
}

/// Extract the text of a chat message's content, which may be a plain string
/// or an OpenAI-style array of parts (e.g. `[{"type":"text","text":...}]`).
fn extract_message_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts.iter()
            .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Port of a loaded model that can serve chat completions, falling back to the
/// legacy default. Speech/image/video models are registered in the same map but
/// do not run an HTTP chat endpoint.
fn first_chat_port(models: &HashMap<String, LoadedModelEntry>) -> u16 {
    models
        .values()
        .find(|e| crate::task_tags::backend_category_for_tags(&e.task_tags).is_none())
        .map(|e| e.port)
        .unwrap_or(50052)
}

/// OpenAI-compatible `/v1/embeddings` — thin passthrough to the loaded text
/// model's own llama-server `/v1/embeddings` endpoint.
async fn embeddings(
    State(state): State<AppState>,
    Json(payload): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, (axum::http::StatusCode, String)> {
    let port: u16 = {
        let models = state.loaded_models.lock().await;
        first_chat_port(&models)
    };

    let inputs: Vec<String> = match payload.input {
        EmbeddingsInput::Single(s) => vec![s],
        EmbeddingsInput::Multiple(v) => v,
    };

    let client = reqwest::Client::builder().no_proxy().build().unwrap_or_else(|_| reqwest::Client::new());
    let url = format!("http://127.0.0.1:{}/v1/embeddings", port);
    let req_body = serde_json::json!({ "model": "local", "input": inputs });

    let res = client.post(&url).json(&req_body).send().await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;

    if !res.status().is_success() {
        let status = res.status();
        let err = res.text().await.unwrap_or_default();
        return Err((axum::http::StatusCode::BAD_GATEWAY, format!("llama-server /v1/embeddings {} error: {}", status, err)));
    }

    let body: serde_json::Value = res.json().await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;

    let data: Vec<EmbeddingObject> = body["data"].as_array()
        .map(|arr| arr.iter().enumerate().map(|(i, item)| {
            let embedding = item["embedding"].as_array()
                .map(|e| e.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect())
                .unwrap_or_default();
            let index = item["index"].as_u64().map(|v| v as usize).unwrap_or(i);
            EmbeddingObject { object: "embedding", embedding, index }
        }).collect())
        .unwrap_or_default();

    let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
    let total_tokens = body["usage"]["total_tokens"].as_u64().unwrap_or(prompt_tokens as u64) as u32;

    Ok(Json(EmbeddingsResponse {
        object: "list",
        data,
        model: payload.model,
        usage: EmbeddingsUsage { prompt_tokens, total_tokens },
    }))
}

async fn stream_chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<StreamChatRequest>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>, (axum::http::StatusCode, String)> {

    // Determine which port to send to
    let port: u16 = {
        let models = state.loaded_models.lock().await;
        if let Some(mid) = &payload.model_id {
            // Route to the requested model
            models.get(mid)
                .map(|e| e.port)
                .unwrap_or(50052)
        } else {
            // Chat needs a model that speaks /v1/chat/completions. Tool calls can
            // load speech or image models alongside it, so never route to those.
            first_chat_port(&models)
        }
    };

    // Auto-load via backend trait if nothing is loaded yet (backwards compat)
    let nothing_loaded = state.loaded_models.lock().await.is_empty();
    if nothing_loaded {
        let backend_arc = state
            .registry
            .get_backend(Modality::Text)
            .await
            .ok_or((axum::http::StatusCode::SERVICE_UNAVAILABLE, "No text backend registered".to_string()))?;
        let is_loaded = { let b = backend_arc.read().await; b.is_loaded() };
        if !is_loaded {
            let model_path = std::path::PathBuf::from(&payload.inner.model);
            let mut b = backend_arc.write().await;
            let load_path = if model_path.exists() { model_path } else { std::path::PathBuf::from("models/default.gguf") };
            let _ = b.load_model(&load_path, &backend_trait::LoadOptions::default()).await;
            let mut status = state.active_model.lock().await;
            *status = LoadedModelStatus {
                is_loaded: true,
                model_path: Some(load_path.to_string_lossy().to_string()),
                gpu_layers: Some(99),
                context_size: Some(4096),
            };
        }
    }

    // Build messages with system prompt. If any message carries an image_url,
    // forward the multimodal content array unchanged to llama-server (which
    // needs --mmproj loaded to actually accept the image part).
    let image_input = extract_image_input(&payload.inner.messages);
    let has_image = image_input.is_some();
    let last_user_idx = if has_image {
        payload.inner.messages.iter().rposition(|m| m.role == "user")
    } else {
        None
    };
    let conversation_id = conversation_key(&payload.inner.conversation_id);
    let image_media_id = if let Some((bytes, mime)) = image_input {
        let id = state.media_store.store(bytes, &mime).await;
        let upload_prompt = payload.inner.messages.iter().rev()
            .find(|m| m.role == "user")
            .map(|m| extract_message_text(&m.content))
            .unwrap_or_default();
        crate::media_store::record_generated_asset(
            &state.generated_assets,
            &conversation_id,
            GeneratedAsset {
                id: id.clone(),
                modality: "image".to_string(),
                model: "user_upload".to_string(),
                prompt: upload_prompt,
                created_at: std::time::Instant::now(),
            },
        ).await;
        Some(id)
    } else {
        None
    };
    let tools_enabled = payload.enable_tools.unwrap_or(true) && payload.inner.enable_tools.unwrap_or(true);
    let tool_schemas = if tools_enabled {
        orchestrator_tools::schemas()
    } else {
        vec![]
    };
    let system_content = if !tool_schemas.is_empty() {
        let fallback_prompt = tool_fallback::build_tool_prompt(&tool_schemas);
        format!(
            "You are a helpful, knowledgeable AI assistant that also orchestrates the other models \
             installed on this machine. You can discover what is available and run any of it — \
             image generation, speech, transcription, video — when the user's request calls for it. \
             Use tools proactively; don't just describe what you could do, actually do it.{}",
            fallback_prompt
        )
    } else {
        "You are a helpful, knowledgeable AI assistant.".to_string()
    };
    let system_content = match &image_media_id {
        Some(id) => format!("{}{}", system_content, image_handle_note(id)),
        None => system_content,
    };
    let system_content = format!(
        "{}{}",
        system_content,
        generated_assets_manifest(&state, &conversation_id).await
    );

    let mut chat_messages: Vec<serde_json::Value> = vec![
        serde_json::json!({ "role": "system", "content": system_content })
    ];
    for (i, msg) in payload.inner.messages.iter().enumerate() {
        if Some(i) == last_user_idx && has_image {
            let content = match &msg.content {
                serde_json::Value::Array(parts) => parts.clone(),
                serde_json::Value::String(s) => vec![serde_json::json!({"type":"text","text":s})],
                _ => vec![],
            };
            chat_messages.push(serde_json::json!({
                "role": msg.role,
                "content": content
            }));
        } else {
            chat_messages.push(serde_json::json!({ "role": msg.role, "content": msg.content }));
        }
    }
    // The user resolved a list_models ambiguity by picking one option. Pin it
    // as a hint rather than assuming the goal was run_model — the original
    // request might have been e.g. "what params does this model take?", which
    // should still resolve to inspect_model.
    if let Some(model) = payload.forced_model.as_ref().filter(|m| !m.is_empty()) {
        chat_messages.push(serde_json::json!({
            "role": "system",
            "content": format!(
                "The user picked '{}' from the list of matching models. Use exactly \
                 this model name for whichever tool call fits their request \
                 (run_model or inspect_model).",
                model
            )
        }));
    }

    let tools_json: Vec<serde_json::Value> = tool_schemas.iter().map(|t| {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
        })
    }).collect();

    let max_tokens = payload.inner.max_tokens.unwrap_or(4096);
    let temperature = payload.inner.temperature.unwrap_or(0.7);
    let top_p = payload.inner.top_p.unwrap_or(0.9);
    let thinking = payload.inner.enable_thinking.unwrap_or(true);

    let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
    let client = reqwest::Client::builder().no_proxy().build().unwrap_or_else(|_| reqwest::Client::new());
    let upstream = client
        .post(&url)
        .json(&stream_request_body(&chat_messages, &tools_json, max_tokens, temperature, top_p, thinking))
        .send()
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;

    let last_user_prompt = payload.inner.messages.iter().rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_str())
        .unwrap_or("")
        .to_string();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(512);

    let autopilot = payload.autopilot.unwrap_or(false);

    let task_state = state.clone();
    let task_tool_schemas = tool_schemas.clone();
    let task_conversation_id = conversation_id.clone();
    tokio::spawn(async move {
        // The orchestrator loop: generate, run whatever tool the model asked for,
        // hand the result back, generate again. Ends on a turn that asks for none.
        let mut messages = chat_messages;
        let mut pending = Some(upstream);

        for hop in 0..MAX_TOOL_HOPS {
            let response = match pending.take() {
                Some(response) => response,
                None => {
                    let body = stream_request_body(&messages, &tools_json, max_tokens, temperature, top_p, thinking);
                    match client.post(&url).json(&body).send().await {
                        Ok(response) => response,
                        Err(e) => {
                            tracing::warn!("follow-up generation after tool result failed: {}", e);
                            break;
                        }
                    }
                }
            };

            let mut byte_stream = response.bytes_stream();
            let mut accumulated_content = String::new();
            let mut accumulated_reasoning = String::new();
            // Native streaming tool calls, keyed by `index`: (id, name, argument fragments)
            let mut native_tool_calls: std::collections::BTreeMap<u64, (String, String, String)> =
                std::collections::BTreeMap::new();

            while let Some(chunk) = byte_stream.next().await {
                let buf = match chunk {
                    Ok(buf) => buf,
                    Err(_) => break,
                };
                let text = String::from_utf8_lossy(&buf).to_string();
                for line in text.lines() {
                    if !line.starts_with("data:") {
                        continue;
                    }
                    let data = line.trim_start_matches("data:").trim();
                    if data.is_empty() {
                        continue;
                    }
                    if data == "[DONE]" {
                        // Don't forward the upstream [DONE] — more hops may follow.
                        // We send our own once the whole exchange is over.
                        continue;
                    }

                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(delta) = json["choices"][0].get("delta") {
                            if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
                                accumulated_content.push_str(c);
                            }
                            if let Some(r) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                                accumulated_reasoning.push_str(r);
                            }
                            if let Some(tcs) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                                for tc in tcs {
                                    let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                                    let slot = native_tool_calls.entry(idx).or_default();
                                    if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                        slot.0 = id.to_string();
                                    }
                                    if let Some(name) = tc.pointer("/function/name").and_then(|v| v.as_str()) {
                                        slot.1 = name.to_string();
                                    }
                                    if let Some(args) = tc.pointer("/function/arguments").and_then(|v| v.as_str()) {
                                        slot.2.push_str(args);
                                    }
                                }
                            }
                        }
                    }

                    if tx.send(Ok(Event::default().data(data.to_string()))).await.is_err() {
                        return;
                    }
                }
            }

            if !tools_enabled || task_tool_schemas.is_empty() {
                break;
            }

            let combined = if !accumulated_reasoning.is_empty() {
                format!("<think>\n{}\n</think>\n\n{}", accumulated_reasoning, accumulated_content)
            } else {
                accumulated_content.clone()
            };

            // A tool-capable chat template makes llama-server emit structured
            // `tool_calls` deltas. Those win — the text fallbacks only exist for
            // templates that can't do that.
            let native_call = native_tool_calls.values().find(|(_, name, _)| !name.is_empty())
                .map(|(id, name, args)| ToolCall {
                    id: if id.is_empty() { format!("native-{}", uuid::Uuid::new_v4()) } else { id.clone() },
                    name: name.clone(),
                    arguments: serde_json::from_str(args.trim())
                        .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                });

            let resolved_call = native_call
                .or_else(|| tool_fallback::parse_tool_calls(&combined).map(|parsed| parsed.tool_call))
                .or_else(|| {
                    // Loose intent matching only on the opening turn: once a tool has
                    // run, the model names it while describing the result, which would
                    // otherwise re-trigger the same call every hop.
                    if hop == 0 {
                        tool_fallback::detect_tool_intent(&combined, &task_tool_schemas, &last_user_prompt)
                    } else {
                        None
                    }
                });

            let Some(tool_call) = resolved_call else {
                break;
            };

            tracing::info!("stream_chat_completions: hop {} calls tool '{}'", hop, tool_call.name);

            let job_id = format!("tool-{}", uuid::Uuid::new_v4());

            let status_event = serde_json::json!({
                "type": "tool_status",
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                "status": "executing",
                "job_id": job_id,
            });
            if tx.send(Ok(Event::default().data(status_event.to_string()))).await.is_err() {
                return;
            }

            // Dispatch via the orchestrator tools (list_models / run_model).
            let result = match orchestrator_tools::dispatch(&task_state, &tool_call, &job_id, &task_conversation_id).await {
                Some(result) => result,
                None => backend_trait::ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content: format!("Unknown tool: '{}'.", tool_call.name),
                    media_handle: None,
                    media_data: None,
                    media_type: None,
                    is_error: true,
                },
            };

            let mut result_event = serde_json::json!({
                "type": "tool_result",
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                "content": result.content,
                "is_error": result.is_error,
                "job_id": job_id,
            });
            if let Some(handle) = &result.media_handle {
                result_event["media_url"] = serde_json::Value::String(handle.clone());
            }
            if let Some(media_type) = &result.media_type {
                result_event["media_type"] = serde_json::Value::String(media_type.clone());
            }
            if tx.send(Ok(Event::default().data(result_event.to_string()))).await.is_err() {
                return;
            }

            // The model was browsing available models. If more than one is compatible
            // with what the user asked for, pause and let the user pick rather than
            // letting the small orchestrator choose (unless autopilot is on).
            if tool_call.name == "list_models" && !autopilot {
                if let Ok(mut rows) = serde_json::from_str::<Vec<serde_json::Value>>(&result.content) {
                    if let Some(target) = infer_modality(&last_user_prompt) {
                        let filtered: Vec<serde_json::Value> = rows.iter()
                            .filter(|row| {
                                let tags: Vec<String> = row["task_tags"].as_array()
                                    .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                                    .unwrap_or_default();
                                crate::task_tags::backend_category_for_tags(&tags) == Some(target)
                            })
                            .cloned()
                            .collect();
                        if !filtered.is_empty() {
                            rows = filtered;
                        }
                    }
                    if rows.len() > 1 {
                        let choice_event = serde_json::json!({
                            "type": "model_choice",
                            "options": rows,
                        });
                        let _ = tx.send(Ok(Event::default().data(choice_event.to_string()))).await;
                        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                        return;
                    }
                }
            }

            // Hand the exchange back so the next hop can act on the result.
            messages.push(serde_json::json!({
                "role": "assistant",
                "content": accumulated_content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments.to_string(),
                    }
                }]
            }));
            let mut feedback = result.content.clone();
            if let Some(ref handle) = result.media_handle {
                feedback.push_str(&format!(
                    " (the media has already been shown to the user) [Media available at: {}]",
                    handle
                ));
            }
            messages.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": feedback,
            }));
        }

        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    let sse_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Sse::new(sse_stream).keep_alive(KeepAlive::default()))
}

/// Best-effort guess of the output modality the user is asking for, so the
/// model picker only offers compatible models. Returns None when unsure.
fn infer_modality(prompt: &str) -> Option<Modality> {
    let p = prompt.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| p.contains(w));
    if has(&["image", "picture", "photo", "draw", "render", "paint", "logo", "wallpaper", "illustration"]) {
        Some(Modality::Image)
    } else if has(&["video", "animate", "animation", "movie", " clip"]) {
        Some(Modality::Video)
    } else if has(&["transcribe", "transcription", "transcript"]) {
        Some(Modality::AudioAsr)
    } else if has(&["speak", "voice", "narrate", "read aloud", "speech", "pronounce", "text to speech", "tts"]) {
        Some(Modality::AudioTts)
    } else {
        None
    }
}

/// One upstream request body for a single generation turn.
fn stream_request_body(
    messages: &[serde_json::Value],
    tools: &[serde_json::Value],
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    thinking: bool,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": 1.15,
        "stream": true,
        "thinking": thinking,
    });
    if !tools.is_empty() {
        body["tools"] = serde_json::Value::Array(tools.to_vec());
    }
    body
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (axum::http::StatusCode, String)> {
    let tools_enabled = payload.enable_tools.unwrap_or(true);

    // Determine port (same logic as streaming path)
    let port: u16 = {
        let models = state.loaded_models.lock().await;
        first_chat_port(&models)
    };

    // Auto-load if nothing loaded (backwards compat)
    let nothing_loaded = state.loaded_models.lock().await.is_empty();
    if nothing_loaded {
        let backend_arc = state.registry.get_backend(Modality::Text).await
            .ok_or((axum::http::StatusCode::SERVICE_UNAVAILABLE, "No text backend registered".into()))?;
        let is_loaded = { backend_arc.read().await.is_loaded() };
        if !is_loaded {
            let model_path = std::path::PathBuf::from(&payload.model);
            let mut b = backend_arc.write().await;
            let load_path = if model_path.exists() { model_path } else { std::path::PathBuf::from("models/default.gguf") };
            let _ = b.load_model(&load_path, &backend_trait::LoadOptions::default()).await;
        }
    }

    // Build messages
    let image_input = extract_image_input(&payload.messages);
    let has_image = image_input.is_some();
    let last_user_idx = if has_image {
        payload.messages.iter().rposition(|m| m.role == "user")
    } else {
        None
    };
    let conversation_id = conversation_key(&payload.conversation_id);
    let image_media_id = if let Some((bytes, mime)) = image_input {
        let id = state.media_store.store(bytes, &mime).await;
        let upload_prompt = payload.messages.iter().rev()
            .find(|m| m.role == "user")
            .map(|m| extract_message_text(&m.content))
            .unwrap_or_default();
        crate::media_store::record_generated_asset(
            &state.generated_assets,
            &conversation_id,
            GeneratedAsset {
                id: id.clone(),
                modality: "image".to_string(),
                model: "user_upload".to_string(),
                prompt: upload_prompt,
                created_at: std::time::Instant::now(),
            },
        ).await;
        Some(id)
    } else {
        None
    };

    // If tools enabled but we want to use prompt-based fallback, append tool instructions to system prompt
    let tool_schemas = if tools_enabled {
        orchestrator_tools::schemas()
    } else {
        vec![]
    };
    let system_content = if !tool_schemas.is_empty() {
        let fallback_prompt = tool_fallback::build_tool_prompt(&tool_schemas);
        format!(
            "You are a helpful, knowledgeable AI assistant that also orchestrates the other models \
             installed on this machine. You can discover what is available and run any of it — \
             image generation, speech, transcription, video — when the user's request calls for it. \
             Use tools proactively; don't just describe what you could do, actually do it.{}",
            fallback_prompt
        )
    } else {
        "You are a helpful, knowledgeable AI assistant.".to_string()
    };
    let system_content = match &image_media_id {
        Some(id) => format!("{}{}", system_content, image_handle_note(id)),
        None => system_content,
    };
    let system_content = format!(
        "{}{}",
        system_content,
        generated_assets_manifest(&state, &conversation_id).await
    );

    let mut chat_messages: Vec<serde_json::Value> = vec![
        serde_json::json!({"role": "system", "content": system_content})
    ];

    for (i, msg) in payload.messages.iter().enumerate() {
        if Some(i) == last_user_idx && has_image {
            let content = match &msg.content {
                serde_json::Value::Array(parts) => parts.clone(),
                serde_json::Value::String(s) => vec![serde_json::json!({"type":"text","text":s})],
                _ => vec![],
            };
            chat_messages.push(serde_json::json!({"role": msg.role, "content": content}));
        } else {
            chat_messages.push(serde_json::json!({"role": msg.role, "content": msg.content}));
        }
    }

    let request_id = format!("chatcmpl-{}", uuid_simple());

    // Build the request body — include tools if enabled
    let mut req_body = serde_json::json!({
        "model": "local",
        "messages": chat_messages,
        "max_tokens": payload.max_tokens.unwrap_or(4096),
        "temperature": payload.temperature.unwrap_or(0.7),
        "top_p": payload.top_p.unwrap_or(0.9),
        "repeat_penalty": 1.15,
        "thinking": payload.enable_thinking.unwrap_or(true),
    });

    // Try native tool calling first
    if !tool_schemas.is_empty() {
        let tools_json: Vec<serde_json::Value> = tool_schemas.iter().map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })
        }).collect();
        req_body["tools"] = serde_json::Value::Array(tools_json);
        if let Some(tc) = &payload.tool_choice {
            req_body["tool_choice"] = serde_json::Value::String(tc.clone());
        }
    }

    let client = reqwest::Client::builder().no_proxy().build().unwrap_or_else(|_| reqwest::Client::new());
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);

    let res = client.post(&url).json(&req_body).send().await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;

    if !res.status().is_success() {
        let status = res.status();
        let err = res.text().await.unwrap_or_default();
        return Err((axum::http::StatusCode::BAD_GATEWAY, format!("llama-server {} error: {}", status, err)));
    }

    let body: serde_json::Value = res.json().await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;

    let msg = &body["choices"][0]["message"];
    let finish_reason = body["choices"][0]["finish_reason"].as_str().unwrap_or("stop");

    // Check for native tool calls in the response
    let native_tool_calls = msg.get("tool_calls").and_then(|tc| tc.as_array()).cloned();

    // Check for prompt-based fallback tool calls in the content
    let content_text = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
    let reasoning = msg.get("reasoning_content").and_then(|r| r.as_str()).unwrap_or("");

    let combined_text = if !reasoning.is_empty() {
        format!("<think>\n{}\n</think>\n\n{}", reasoning, content_text)
    } else {
        content_text.to_string()
    };

    if let Some(ref tool_calls_json) = native_tool_calls {
        if !tool_calls_json.is_empty() {
            // Dispatch tool calls
            let mut tool_results = Vec::new();
            for tc_json in tool_calls_json {
                let tc = ToolCall {
                    id: tc_json["id"].as_str().unwrap_or("").to_string(),
                    name: tc_json["function"]["name"].as_str().unwrap_or("").to_string(),
                    arguments: serde_json::from_str(
                        tc_json["function"]["arguments"].as_str().unwrap_or("{}")
                    ).unwrap_or(serde_json::json!({})),
                };
                let job_id = format!("tool-{}", uuid::Uuid::new_v4());
                let result = match orchestrator_tools::dispatch(&state, &tc, &job_id, &conversation_id).await {
                    Some(result) => result,
                    None => backend_trait::ToolResult {
                        tool_call_id: tc.id.clone(),
                        content: format!("Unknown tool: '{}'.", tc.name),
                        media_handle: None,
                        media_data: None,
                        media_type: None,
                        is_error: true,
                    },
                };
                tool_results.push((tc, result));
            }

            // Re-call the model with tool results
            for (tc, result) in &tool_results {
                chat_messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default(),
                        }
                    }]
                }));
                let tool_content = if let Some(ref handle) = result.media_handle {
                    format!("{}\n[Media available at: {}]", result.content, handle)
                } else {
                    result.content.clone()
                };
                chat_messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_content,
                }));
            }

            // Second call to model (without tools to prevent chaining)
            let followup_body = serde_json::json!({
                "model": "local",
                "messages": chat_messages,
                "max_tokens": payload.max_tokens.unwrap_or(4096),
                "temperature": payload.temperature.unwrap_or(0.7),
                "top_p": payload.top_p.unwrap_or(0.9),
                "repeat_penalty": 1.15,
                "thinking": payload.enable_thinking.unwrap_or(true),
            });

            let followup_res = client.post(&url).json(&followup_body).send().await
                .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;
            let followup_body: serde_json::Value = followup_res.json().await
                .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;

            let followup_msg = &followup_body["choices"][0]["message"];
            let followup_text = followup_msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            let followup_reasoning = followup_msg.get("reasoning_content").and_then(|r| r.as_str()).unwrap_or("");
            let final_text = if !followup_reasoning.is_empty() {
                format!("<think>\n{}\n</think>\n\n{}", followup_reasoning, followup_text)
            } else {
                followup_text.to_string()
            };

            return Ok(Json(ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".into(),
                created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
                model: payload.model,
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: ChatCompletionMessage {
                        role: "assistant".into(),
                        content: serde_json::Value::String(final_text),
                        tool_calls: None,
                    },
                    finish_reason: "stop".into(),
                }],
            }));
        }
    }

    // Check prompt-based fallback if native tools weren't used or didn't produce calls
    if tools_enabled && !tool_schemas.is_empty() && native_tool_calls.is_none() {
        let last_user_prompt = payload.messages.iter().rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_str())
            .unwrap_or("");

        let resolved_call = tool_fallback::parse_tool_calls(&combined_text)
            .map(|parsed| (parsed.text_before, parsed.tool_call))
            .or_else(|| {
                tool_fallback::detect_tool_intent(&combined_text, &tool_schemas, last_user_prompt)
                    .map(|tc| (combined_text.clone(), tc))
            });

        if let Some((text_before, tool_call)) = resolved_call {
            let job_id = format!("tool-{}", uuid::Uuid::new_v4());
            let result = match orchestrator_tools::dispatch(&state, &tool_call, &job_id, &conversation_id).await {
                Some(result) => result,
                None => backend_trait::ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content: format!("Unknown tool: '{}'.", tool_call.name),
                    media_handle: None,
                    media_data: None,
                    media_type: None,
                    is_error: true,
                },
            };
            let tool_content = if let Some(ref handle) = result.media_handle {
                format!("{}\n[Media available at: {}]", result.content, handle)
            } else {
                result.content.clone()
            };

            // Re-call with tool result appended
            chat_messages.push(serde_json::json!({"role": "assistant", "content": text_before}));
            chat_messages.push(serde_json::json!({
                "role": "user",
                "content": format!("Tool '{}' returned: {}", tool_call.name, tool_content),
            }));

            let followup_body = serde_json::json!({
                "model": "local",
                "messages": chat_messages,
                "max_tokens": payload.max_tokens.unwrap_or(4096),
                "temperature": payload.temperature.unwrap_or(0.7),
                "top_p": payload.top_p.unwrap_or(0.9),
                "repeat_penalty": 1.15,
                "thinking": payload.enable_thinking.unwrap_or(true),
            });

            let followup_res = client.post(&url).json(&followup_body).send().await
                .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;
            let followup_body: serde_json::Value = followup_res.json().await
                .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;

            let followup_msg = &followup_body["choices"][0]["message"];
            let followup_text = followup_msg.get("content").and_then(|c| c.as_str()).unwrap_or("");

            return Ok(Json(ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".into(),
                created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
                model: payload.model,
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: ChatCompletionMessage {
                        role: "assistant".into(),
                        content: serde_json::Value::String(followup_text.to_string()),
                        tool_calls: None,
                    },
                    finish_reason: "stop".into(),
                }],
            }));
        }
    }

    // No tool calls — return the normal response
    Ok(Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".into(),
        created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
        model: payload.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".into(),
                content: serde_json::Value::String(combined_text),
                tool_calls: None,
            },
            finish_reason: finish_reason.to_string(),
        }],
    }))
}

/// Insert a fresh "queued" progress record for `job_id` into the shared map.
pub(crate) async fn init_job_progress(state: &AppState, job_id: &str, modality: &str) {
    let mut jobs = state.job_progress.lock().await;
    jobs.insert(
        job_id.to_string(),
        GenerationProgress {
            job_id: job_id.to_string(),
            modality: modality.to_string(),
            phase: "queued".to_string(),
            step: 0,
            total: 0,
            percent: -1.0,
            status: "running".to_string(),
            message: None,
            updated_at: now_millis(),
        },
    );
}

/// Spawn a background task that polls `backend.poll_progress()` every 150ms
/// and writes results into the shared job map, stamping `job_id`/`modality`.
/// Stops once `done_flag` is set.
pub(crate) fn spawn_progress_poller(
    jobs_map: Arc<tokio::sync::Mutex<HashMap<String, GenerationProgress>>>,
    backend_arc: Arc<tokio::sync::RwLock<Box<dyn InferenceBackend>>>,
    job_id: String,
    modality: &'static str,
    done_flag: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(150));
        loop {
            interval.tick().await;
            if done_flag.load(AtomicOrdering::Relaxed) {
                break;
            }
            let progress = {
                let b = backend_arc.read().await;
                b.poll_progress().await
            };
            if let Some(mut p) = progress {
                p.job_id = job_id.clone();
                p.modality = modality.to_string();
                let mut jobs = jobs_map.lock().await;
                jobs.insert(job_id.clone(), p);
            }
        }
    })
}

/// Finalize a job's progress record as done or errored. A `error` message
/// exactly matching `BackendError::Cancelled`'s Display text is reported as
/// a "cancelled" status/phase rather than a generic "error", so the frontend
/// can distinguish a user-initiated cancellation from a real failure.
pub(crate) async fn finalize_job_progress(state: &AppState, job_id: &str, modality: &str, error: Option<&str>) {
    let cancelled = error == Some("Generation cancelled");
    let status = if cancelled { "cancelled" } else if error.is_some() { "error" } else { "done" };
    let mut jobs = state.job_progress.lock().await;
    jobs.insert(
        job_id.to_string(),
        GenerationProgress {
            job_id: job_id.to_string(),
            modality: modality.to_string(),
            phase: status.to_string(),
            step: 0,
            total: 0,
            percent: if error.is_some() { -1.0 } else { 100.0 },
            status: status.to_string(),
            message: error.map(|e| e.to_string()),
            updated_at: now_millis(),
        },
    );
}

/// Error type for `generate_images`, so a `BackendError::MissingComponents`
/// can be surfaced as a structured 422 JSON body (for the frontend's
/// "download missing components" fix) while every other error path keeps the
/// existing plain-text response shape.
enum GenerateImagesError {
    Simple(axum::http::StatusCode, String),
    MissingComponents(Vec<backend_trait::MissingComponentInfo>),
}

impl axum::response::IntoResponse for GenerateImagesError {
    fn into_response(self) -> axum::response::Response {
        match self {
            GenerateImagesError::Simple(status, msg) => (status, msg).into_response(),
            GenerateImagesError::MissingComponents(list) => {
                let missing_components: Vec<serde_json::Value> = list
                    .into_iter()
                    .map(|c| {
                        serde_json::json!({
                            "kind_name": c.kind_name,
                            "target_path": c.target_path,
                            "source": c.source.map(|s| serde_json::json!({
                                "repo": s.repo,
                                "filename": s.filename,
                                "target_filename": s.target_filename,
                            })),
                        })
                    })
                    .collect();
                let body = serde_json::json!({
                    "error": "Missing required components",
                    "missing_components": missing_components,
                });
                (axum::http::StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
            }
        }
    }
}

async fn generate_images(
    State(state): State<AppState>,
    Json(payload): Json<ImageGenerationRequest>,
) -> Result<Json<ImageGenerationResponse>, GenerateImagesError> {
    use base64::Engine;
    let backend_arc = state
        .registry
        .get_backend(Modality::Image)
        .await
        .ok_or(GenerateImagesError::Simple(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No image backend registered".to_string(),
        ))?;

    let job_id = payload.job_id.clone().unwrap_or_else(|| format!("job-{}", uuid_simple()));
    let request_id = format!("img-{}", uuid_simple());

    // OpenAI-style "WxH" size string is a fallback when width/height aren't
    // given explicitly.
    let (size_w, size_h) = payload
        .size
        .as_deref()
        .and_then(|s| s.split_once('x'))
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .map(|(w, h)| (Some(w), Some(h)))
        .unwrap_or((None, None));

    let image_params = backend_trait::ImageParams {
        negative_prompt: payload.negative_prompt,
        steps: payload.steps,
        cfg_scale: payload.cfg_scale,
        width: payload.width.or(size_w),
        height: payload.height.or(size_h),
        // A negative seed is the UI's "-1 = random" sentinel. Normalize it to
        // None so backends generate a fresh random seed instead of pinning to
        // the literal value (sd.exe treats -1 as random, but diffusers'
        // manual_seed(-1) would lock every render to one fixed seed).
        seed: payload.seed.filter(|s| *s >= 0),
        strength: payload.strength,
    };

    let image_input = payload
        .image
        .as_deref()
        .and_then(|b64| base64::engine::general_purpose::STANDARD.decode(b64).ok());

    let inf_req = InferenceRequest {
        request_id,
        prompt: payload.prompt,
        messages: None,
        sampling: SamplingParams::default(),
        modality: Modality::Image,
        image_input,
        tools: None,
        tool_choice: None,
        image_params: Some(image_params),
        mesh_params: None,
        audio_params: None,
        video_params: None,
        cancel: None,
    };

    {
        let is_loaded = {
            let b = backend_arc.read().await;
            b.is_loaded()
        };
        if !is_loaded {
            let mut b = backend_arc.write().await;
            let _ = b.load_model(std::path::Path::new("models/sd.safetensors"), &backend_trait::LoadOptions::default()).await;
        }
    }

    init_job_progress(&state, &job_id, "image").await;
    let done_flag = Arc::new(AtomicBool::new(false));
    let poller = spawn_progress_poller(
        state.job_progress.clone(),
        backend_arc.clone(),
        job_id.clone(),
        "image",
        done_flag.clone(),
    );

    let cancel_flag = Arc::new(AtomicBool::new(false));
    state.job_cancel.lock().await.insert(job_id.clone(), cancel_flag.clone());
    let mut inf_req = inf_req;
    inf_req.cancel = Some(cancel_flag);

    let backend = backend_arc.read().await;
    let inf_res = backend.generate(inf_req).await;
    done_flag.store(true, AtomicOrdering::Relaxed);
    poller.abort();
    state.job_cancel.lock().await.remove(&job_id);

    let inf_res = match inf_res {
        Ok(res) => res,
        Err(e) => {
            finalize_job_progress(&state, &job_id, "image", Some(&e.to_string())).await;
            return Err(match e {
                backend_trait::BackendError::MissingComponents(list) => {
                    GenerateImagesError::MissingComponents(list)
                }
                other => GenerateImagesError::Simple(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    other.to_string(),
                ),
            });
        }
    };
    finalize_job_progress(&state, &job_id, "image", None).await;

    let img_bytes = inf_res.output_data.unwrap_or_default();
    let b64_str = use_base64_encode(&img_bytes);

    Ok(Json(ImageGenerationResponse {
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        data: vec![ImageObject { b64_json: b64_str }],
        job_id,
    }))
}

async fn generate_mesh3d(
    State(state): State<AppState>,
    Json(payload): Json<Mesh3dGenerationRequest>,
) -> Result<Json<Mesh3dGenerationResponse>, (axum::http::StatusCode, String)> {
    use base64::Engine;

    let backend_arc = state
        .registry
        .get_backend(Modality::Mesh3D)
        .await
        .ok_or((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No 3D backend registered".to_string(),
        ))?;

    let is_loaded = {
        let b = backend_arc.read().await;
        b.is_loaded()
    };
    if !is_loaded {
        return Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No 3D model loaded".to_string(),
        ));
    }

    let job_id = payload.job_id.clone().unwrap_or_else(|| format!("job-{}", uuid_simple()));
    let request_id = format!("mesh3d-{}", uuid_simple());

    let images: Vec<Vec<u8>> = payload
        .images
        .clone()
        .unwrap_or_default()
        .iter()
        .map(|b64| base64::engine::general_purpose::STANDARD.decode(b64))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("Invalid base64 image: {}", e)))?;

    let mesh_params = backend_trait::Mesh3dParams {
        input_kind: Some(payload.input_kind.clone()),
        images: if images.is_empty() { None } else { Some(images) },
        steps: payload.steps,
        guidance_scale: payload.guidance_scale,
        seed: payload.seed,
        output_format: payload.output_format.clone(),
        texture: payload.texture,
        foreground_ratio: payload.foreground_ratio,
        extra: payload.extra.clone(),
    };

    let inf_req = InferenceRequest {
        request_id,
        prompt: payload.prompt.clone().unwrap_or_default(),
        messages: None,
        sampling: SamplingParams::default(),
        modality: Modality::Mesh3D,
        image_input: None,
        tools: None,
        tool_choice: None,
        image_params: None,
        mesh_params: Some(mesh_params),
        audio_params: None,
        video_params: None,
        cancel: None,
    };

    init_job_progress(&state, &job_id, "mesh").await;
    let done_flag = Arc::new(AtomicBool::new(false));
    let poller = spawn_progress_poller(
        state.job_progress.clone(),
        backend_arc.clone(),
        job_id.clone(),
        "mesh",
        done_flag.clone(),
    );

    let cancel_flag = Arc::new(AtomicBool::new(false));
    state.job_cancel.lock().await.insert(job_id.clone(), cancel_flag.clone());
    let mut inf_req = inf_req;
    inf_req.cancel = Some(cancel_flag);

    let backend = backend_arc.read().await;
    let inf_res = backend.generate(inf_req).await;
    done_flag.store(true, AtomicOrdering::Relaxed);
    poller.abort();
    state.job_cancel.lock().await.remove(&job_id);

    let inf_res = match inf_res {
        Ok(res) => res,
        Err(e) => {
            finalize_job_progress(&state, &job_id, "mesh", Some(&e.to_string())).await;
            return Err((axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };
    finalize_job_progress(&state, &job_id, "mesh", None).await;

    let mesh_bytes = inf_res.output_data.unwrap_or_default();
    let mesh_base64 = use_base64_encode(&mesh_bytes);
    let format = if !inf_res.output_text.is_empty() {
        inf_res.output_text
    } else {
        payload.output_format.unwrap_or_else(|| "glb".to_string())
    };

    Ok(Json(Mesh3dGenerationResponse {
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        format,
        mesh_base64,
        generation_time_ms: inf_res.generation_time_ms,
        job_id,
    }))
}

async fn generate_videos(
    State(state): State<AppState>,
    Json(payload): Json<VideoGenerationRequest>,
) -> Result<Json<VideoGenerationResponse>, (axum::http::StatusCode, String)> {
    use base64::Engine;

    let backend_arc = state
        .registry
        .get_backend(Modality::Video)
        .await
        .ok_or((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No video backend registered".to_string(),
        ))?;

    let is_loaded = {
        let b = backend_arc.read().await;
        b.is_loaded()
    };
    if !is_loaded {
        return Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No video model loaded".to_string(),
        ));
    }

    let job_id = payload.job_id.clone().unwrap_or_else(|| format!("job-{}", uuid_simple()));
    let request_id = format!("video-{}", uuid_simple());

    let image = payload
        .image
        .as_ref()
        .map(|b64| base64::engine::general_purpose::STANDARD.decode(b64))
        .transpose()
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("Invalid base64 image: {}", e)))?;

    let video_params = backend_trait::VideoParams {
        negative_prompt: payload.negative_prompt.clone(),
        image,
        num_frames: payload.num_frames,
        height: payload.height,
        width: payload.width,
        num_inference_steps: payload.num_inference_steps,
        guidance_scale: payload.guidance_scale,
        fps: payload.fps,
        seed: payload.seed,
        output_format: payload.output_format.clone(),
        extra: payload.extra.clone(),
    };

    let inf_req = InferenceRequest {
        request_id,
        prompt: payload.prompt.clone(),
        messages: None,
        sampling: SamplingParams::default(),
        modality: Modality::Video,
        image_input: None,
        tools: None,
        tool_choice: None,
        image_params: None,
        mesh_params: None,
        audio_params: None,
        video_params: Some(video_params),
        cancel: None,
    };

    init_job_progress(&state, &job_id, "video").await;
    let done_flag = Arc::new(AtomicBool::new(false));
    let poller = spawn_progress_poller(
        state.job_progress.clone(),
        backend_arc.clone(),
        job_id.clone(),
        "video",
        done_flag.clone(),
    );

    let cancel_flag = Arc::new(AtomicBool::new(false));
    state.job_cancel.lock().await.insert(job_id.clone(), cancel_flag.clone());
    let mut inf_req = inf_req;
    inf_req.cancel = Some(cancel_flag);

    let backend = backend_arc.read().await;
    let inf_res = backend.generate(inf_req).await;
    done_flag.store(true, AtomicOrdering::Relaxed);
    poller.abort();
    state.job_cancel.lock().await.remove(&job_id);

    let inf_res = match inf_res {
        Ok(res) => res,
        Err(e) => {
            finalize_job_progress(&state, &job_id, "video", Some(&e.to_string())).await;
            return Err((axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };
    finalize_job_progress(&state, &job_id, "video", None).await;

    let video_bytes = inf_res.output_data.unwrap_or_default();
    let video_b64 = use_base64_encode(&video_bytes);

    Ok(Json(VideoGenerationResponse {
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        video_b64,
        generation_time_ms: inf_res.generation_time_ms,
        job_id,
    }))
}

/// Proxies `GET /schema` from the running `threed_server.py` process, exposing
/// each 3D adapter's declared parameter spec so a future UI can build a
/// dynamic parameter form.
async fn get_mesh3d_schema(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let backend_arc = state
        .registry
        .get_backend(Modality::Mesh3D)
        .await
        .ok_or((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No 3D backend registered".to_string(),
        ))?;

    let backend = backend_arc.read().await;
    match backend.get_param_schema().await {
        Some(schema) => Ok(Json(schema)),
        None => Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "3D backend has no schema available (not loaded or running in simulation mode)".to_string(),
        )),
    }
}

/// Proxies `GET /schema` from the running `video_diffusers_server.py`
/// process, exposing the loaded pipeline's actually-accepted params so the
/// UI can build a dynamic parameter form. Mirrors `get_mesh3d_schema`.
async fn get_video_schema(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let backend_arc = state
        .registry
        .get_backend(Modality::Video)
        .await
        .ok_or((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No video backend registered".to_string(),
        ))?;

    let backend = backend_arc.read().await;
    match backend.get_param_schema().await {
        Some(schema) => Ok(Json(schema)),
        None => Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "Video backend has no schema available (not loaded or running in simulation mode)".to_string(),
        )),
    }
}

async fn transcribe_audio(
    State(state): State<AppState>,
    Json(payload): Json<TranscriptionRequest>,
) -> Result<Json<TranscriptionResponse>, (axum::http::StatusCode, String)> {
    let backend_arc = state
        .registry
        .get_backend(Modality::AudioAsr)
        .await
        .ok_or((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No audio ASR backend registered".to_string(),
        ))?;

    use base64::Engine;
    let audio_bytes = base64::engine::general_purpose::STANDARD
        .decode(&payload.audio_b64)
        .map_err(|e| {
            (
                axum::http::StatusCode::BAD_REQUEST,
                format!("Invalid audio_b64: {}", e),
            )
        })?;

    let request_id = format!("asr-{}", uuid_simple());

    let inf_req = InferenceRequest {
        request_id,
        prompt: "Transcribe audio input".to_string(),
        messages: None,
        sampling: SamplingParams::default(),
        modality: Modality::AudioAsr,
        image_input: Some(audio_bytes),
        tools: None,
        tool_choice: None,
        image_params: None,
        mesh_params: None,
        audio_params: None,
        video_params: None,
        cancel: None,
    };

    {
        let is_loaded = {
            let b = backend_arc.read().await;
            b.is_loaded()
        };
        if !is_loaded {
            let mut b = backend_arc.write().await;
            let _ = b.load_model(std::path::Path::new("models/whisper.bin"), &backend_trait::LoadOptions::default()).await;
        }
    }

    let backend = backend_arc.read().await;
    let inf_res = backend.generate(inf_req).await.map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
        )
    })?;

    Ok(Json(TranscriptionResponse {
        text: inf_res.output_text,
    }))
}

async fn synthesize_speech(
    State(state): State<AppState>,
    Json(payload): Json<SpeechRequest>,
) -> Result<Json<SpeechResponse>, (axum::http::StatusCode, String)> {
    let backend_arc = state
        .registry
        .get_backend(Modality::AudioTts)
        .await
        .ok_or((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No TTS backend registered".to_string(),
        ))?;

    let job_id = payload.job_id.clone().unwrap_or_else(|| format!("job-{}", uuid_simple()));
    let request_id = format!("tts-{}", uuid_simple());

    let inf_req = InferenceRequest {
        request_id,
        prompt: payload.input,
        messages: None,
        sampling: SamplingParams::default(),
        modality: Modality::AudioTts,
        image_input: None,
        tools: None,
        tool_choice: None,
        image_params: None,
        mesh_params: None,
        audio_params: Some(backend_trait::AudioParams { speed: payload.speed }),
        video_params: None,
        cancel: None,
    };

    {
        let is_loaded = {
            let b = backend_arc.read().await;
            b.is_loaded()
        };
        if !is_loaded {
            let tts_model_path = {
                let models = state.loaded_models.lock().await;
                models
                    .values()
                    .find(|entry| crate::task_tags::backend_category_for_tags(&entry.task_tags) == Some(Modality::AudioTts))
                    .map(|entry| entry.model_path.clone())
            };
            let Some(tts_model_path) = tts_model_path else {
                return Err((
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "No TTS model loaded. Load a TTS model first.".to_string(),
                ));
            };
            let mut b = backend_arc.write().await;
            let _ = b
                .load_model(std::path::Path::new(&tts_model_path), &backend_trait::LoadOptions::default())
                .await;
        }
    }

    init_job_progress(&state, &job_id, "tts").await;

    let cancel_flag = Arc::new(AtomicBool::new(false));
    state.job_cancel.lock().await.insert(job_id.clone(), cancel_flag.clone());
    let mut inf_req = inf_req;
    inf_req.cancel = Some(cancel_flag);

    let backend = backend_arc.read().await;
    let inf_res = backend.generate(inf_req).await;
    state.job_cancel.lock().await.remove(&job_id);

    let inf_res = match inf_res {
        Ok(res) => res,
        Err(e) => {
            finalize_job_progress(&state, &job_id, "tts", Some(&e.to_string())).await;
            return Err((axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };
    finalize_job_progress(&state, &job_id, "tts", None).await;

    let audio_bytes = inf_res.output_data.unwrap_or_default();
    let b64_str = use_base64_encode(&audio_bytes);

    Ok(Json(SpeechResponse { audio_b64: b64_str, job_id }))
}

fn use_base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    let mut i = 0;
    while i < input.len() {
        let b0 = input[i] as u32;
        let b1 = if i + 1 < input.len() { input[i + 1] as u32 } else { 0 };
        let b2 = if i + 2 < input.len() { input[i + 2] as u32 } else { 0 };

        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(ALPHABET[((triple >> 18) & 63) as usize] as char);
        result.push(ALPHABET[((triple >> 12) & 63) as usize] as char);
        if i + 1 < input.len() {
            result.push(ALPHABET[((triple >> 6) & 63) as usize] as char);
        } else {
            result.push('=');
        }
        if i + 2 < input.len() {
            result.push(ALPHABET[(triple & 63) as usize] as char);
        } else {
            result.push('=');
        }

        i += 3;
    }
    result
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", nanos)
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

async fn list_jobs(State(state): State<AppState>) -> Json<Vec<GenerationProgress>> {
    let jobs = state.job_progress.lock().await;
    Json(jobs.values().cloned().collect())
}

async fn get_job_progress(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<GenerationProgress>, axum::http::StatusCode> {
    let jobs = state.job_progress.lock().await;
    jobs.get(&id)
        .cloned()
        .map(Json)
        .ok_or(axum::http::StatusCode::NOT_FOUND)
}

/// Signal cancellation of an in-flight generation job (image/mesh/tts, either
/// started directly via `/v1/images|models3d|audio` or by the orchestrator's
/// `run_model` tool). Sets the job's cancel flag; the backend polls it and
/// aborts the in-flight `sd.exe` process / HTTP request cooperatively.
async fn cancel_job(
    State(state): State<AppState>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    let cancel_map = state.job_cancel.lock().await;
    match cancel_map.get(&job_id) {
        Some(flag) => {
            flag.store(true, AtomicOrdering::Relaxed);
            (axum::http::StatusCode::OK, Json(serde_json::json!({ "ok": true })))
        }
        None => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "ok": false, "error": "job not found" })),
        ),
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelLoadRequest {
    pub model_path: String,
    pub gpu_layers: Option<u32>,
    pub context_size: Option<u32>,
    pub mmproj_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StudioModelsRequest {
    pub paths: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MediaRetentionConfig {
    pub ttl_seconds: u64,
    pub persist_disk: bool,
}

#[derive(Debug, Deserialize)]
pub struct ModelUnloadRequest {
    /// model_id returned by /v1/model/load. If omitted, unloads the single legacy model.
    pub model_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DetectedModelEntry {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    /// `None` means this GGUF does not expose a readable context limit.
    pub max_context_size: Option<u32>,
    /// HuggingFace-style task tag(s) inferred from the model's HF metadata (if
    /// known) or its embedded metadata/file type. May contain more than one tag.
    pub task_tags: Vec<String>,
    /// A vision GGUF also needs a local image projector before chat can accept images.
    pub image_input_available: bool,
    pub mmproj_path: Option<String>,
    /// For mesh3d models (task_tags containing "text-to-3d"/"image-to-3d"),
    /// which input kinds this model accepts (subset of "text", "image",
    /// "multi_image"). `None` for other models.
    pub mesh_input_kinds: Option<Vec<String>>,
    /// For mesh3d models with multi-component repos (e.g. Hunyuan3D):
    /// path to the VAE component subfolder.
    pub mesh_vae_path: Option<String>,
    /// Path to the texture/paint component subfolder.
    pub mesh_texgen_path: Option<String>,
    /// For split-checkpoint image models (e.g. FLUX.2 GGUF, Z-Image): the
    /// resolved (or missing, with a download source if known) sibling
    /// text-encoder/VAE components. `None` for models that aren't
    /// split-checkpoint diffusion-model files.
    pub image_components: Option<Vec<sd_backend::ImageComponentStatus>>,
}

fn find_sibling_mmproj(path: &std::path::Path) -> Option<std::path::PathBuf> {
    let directory = path.parent()?;
    std::fs::read_dir(directory).ok().into_iter().flatten().flatten().find_map(|entry| {
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if name.contains("mmproj") && name.ends_with(".gguf") {
            Some(entry.path())
        } else {
            None
        }
    })
}

fn has_image_projector(path: &std::path::Path) -> bool {
    find_sibling_mmproj(path).is_some()
}

/// For multi-component mesh3d repos (e.g. Hunyuan3D-2.x), detect sibling
/// subfolders that contain the VAE and paint/texture pipeline components.
fn find_mesh3d_siblings(path: &std::path::Path) -> (Option<String>, Option<String>) {
    // path is the repo root directory for mesh3d models
    let dir = if path.is_dir() { path } else { return (None, None); };
    let mut vae_path = None;
    let mut texgen_path = None;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            if !entry.path().is_dir() { continue; }
            if name.contains("vae") {
                vae_path = Some(entry.path().to_string_lossy().to_string());
            } else if name.contains("paint") || name.contains("texgen") || name.contains("texture") {
                texgen_path = Some(entry.path().to_string_lossy().to_string());
            }
        }
    }
    (vae_path, texgen_path)
}

/// A `.safetensors` model is a text LLM if the sibling `config.json` in the
/// same directory declares a `ForCausalLM` architecture (vLLM-style
/// config-driven detection), e.g. `LlamaForCausalLM`, `Qwen2ForCausalLM`.
pub(crate) fn detect_safetensors_causal_lm(path: &std::path::Path) -> bool {
    let Some(directory) = path.parent() else { return false };
    let config_path = directory.join("config.json");
    let Ok(contents) = std::fs::read_to_string(&config_path) else { return false };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&contents) else { return false };
    config
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|architectures| {
            architectures
                .iter()
                .filter_map(|a| a.as_str())
                .any(|a| a.ends_with("ForCausalLM"))
        })
        .unwrap_or(false)
}

/// Tokens that identify a 3D mesh-generator model, matched case-insensitively
/// against the combined "parent dir name + file name" string.
const MESH3D_NAME_TOKENS: &[&str] = &[
    "triposr", "tripo", "shap-e", "shap_e", "shape", "point-e", "point_e", "pointe",
    "instantmesh", "instant-mesh", "instant_mesh", "trellis", "hunyuan3d", "hunyuan-3d",
    "hunyuan_3d", "sf3d", "stable-fast-3d", "stable_fast_3d", "zero123", "zero-1-2-3",
    "wonder3d", "sv3d", "lgm", "crm", "-3d", "_3d", " 3d", "llama-mesh", "llama_mesh", "llamamesh",
];

pub(crate) fn is_mesh3d_name(file_name: &str) -> bool {
    MESH3D_NAME_TOKENS.iter().any(|token| file_name.contains(token))
}

/// Which input kinds a detected 3D mesh-generator model accepts, keyed off
/// the same "parent dir name + file name" string used by `is_mesh3d_name`.
pub(crate) fn detect_mesh_input_kinds(path: &std::path::Path) -> Option<Vec<String>> {
    let file_name = {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let parent = path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or_default();
        format!("{parent} {name}").to_ascii_lowercase()
    };

    if !is_mesh3d_name(&file_name) {
        return None;
    }

    let kinds = if file_name.contains("shap-e") || file_name.contains("shap_e") || file_name.contains("shape")
        || file_name.contains("point-e") || file_name.contains("point_e") || file_name.contains("pointe")
    {
        vec!["text".to_string(), "image".to_string()]
    } else if file_name.contains("triposr") || file_name.contains("tripo") || file_name.contains("sf3d")
        || file_name.contains("stable-fast-3d") || file_name.contains("stable_fast_3d")
    {
        vec!["image".to_string()]
    } else if file_name.contains("instantmesh") || file_name.contains("instant-mesh") || file_name.contains("instant_mesh")
        || file_name.contains("wonder3d") || file_name.contains("sv3d") || file_name.contains("zero123")
        || file_name.contains("zero-1-2-3")
    {
        vec!["image".to_string(), "multi_image".to_string()]
    } else if file_name.contains("trellis") || file_name.contains("hunyuan3d") || file_name.contains("hunyuan-3d")
        || file_name.contains("hunyuan_3d")
    {
        vec!["text".to_string(), "image".to_string(), "multi_image".to_string()]
    } else if file_name.contains("crm") || file_name.contains("lgm") {
        vec!["image".to_string(), "multi_image".to_string()]
    } else if file_name.contains("llama-mesh") || file_name.contains("llama_mesh") || file_name.contains("llamamesh") {
        vec!["text".to_string()]
    } else {
        vec!["image".to_string()]
    };

    Some(kinds)
}

fn read_u32(bytes: &[u8], position: &mut usize) -> Option<u32> {
    let value = bytes.get(*position..*position + 4)?;
    *position += 4;
    Some(u32::from_le_bytes(value.try_into().ok()?))
}

fn read_u64(bytes: &[u8], position: &mut usize) -> Option<u64> {
    let value = bytes.get(*position..*position + 8)?;
    *position += 8;
    Some(u64::from_le_bytes(value.try_into().ok()?))
}

fn skip_gguf_value(bytes: &[u8], position: &mut usize, value_type: u32) -> Option<()> {
    let length = match value_type {
        0 | 1 | 7 => 1, 2 | 3 => 2, 4 | 5 | 6 => 4, 10 | 11 | 12 => 8,
        8 => read_u64(bytes, position)? as usize,
        9 => {
            let item_type = read_u32(bytes, position)?;
            let count = read_u64(bytes, position)? as usize;
            let item_size = match item_type { 0 | 1 | 7 => 1, 2 | 3 => 2, 4 | 5 | 6 => 4, 10 | 11 | 12 => 8, _ => return None };
            count.checked_mul(item_size)?
        }
        _ => return None,
    };
    *position = position.checked_add(length)?;
    (bytes.len() >= *position).then_some(())
}

/// Read the architecture context limit from GGUF key-value metadata.
fn gguf_context_length(path: &std::path::Path) -> Option<u32> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path).ok()?.take(8 * 1024 * 1024).read_to_end(&mut bytes).ok()?;
    if bytes.get(0..4)? != b"GGUF" { return None; }

    // Metadata arrays (notably arrays of strings) are allowed before the
    // architecture fields. Locate the unambiguous key directly first instead
    // of relying on every preceding metadata type being understood.
    let context_key = b".context_length";
    if let Some(key_start) = bytes.windows(context_key.len()).position(|window| window == context_key) {
        let mut value_position = key_start + context_key.len();
        let value_type = read_u32(&bytes, &mut value_position)?;
        return match value_type {
            4 => read_u32(&bytes, &mut value_position),
            5 => read_u32(&bytes, &mut value_position).map(|value| value as i32).filter(|value| *value > 0).map(|value| value as u32),
            10 | 11 => read_u64(&bytes, &mut value_position).and_then(|value| u32::try_from(value).ok()),
            _ => None,
        };
    }

    let mut position = 4;
    let version = read_u32(&bytes, &mut position)?;
    if !(2..=3).contains(&version) { return None; }
    let _tensor_count = read_u64(&bytes, &mut position)?;
    let kv_count = read_u64(&bytes, &mut position)?;
    for _ in 0..kv_count {
        let key_len = read_u64(&bytes, &mut position)? as usize;
        let key = std::str::from_utf8(bytes.get(position..position.checked_add(key_len)?)?).ok()?;
        position += key_len;
        let value_type = read_u32(&bytes, &mut position)?;
        if key.ends_with(".context_length") && matches!(value_type, 4 | 5 | 10 | 11) {
            return match value_type {
                4 => read_u32(&bytes, &mut position),
                5 => read_u32(&bytes, &mut position).map(|value| value as i32).filter(|value| *value > 0).map(|value| value as u32),
                10 | 11 => read_u64(&bytes, &mut position).and_then(|value| u32::try_from(value).ok()),
                _ => None,
            };
        }
        skip_gguf_value(&bytes, &mut position, value_type)?;
    }
    None
}

/// Sum the byte sizes of model weight files inside `dir`, recursing into
/// component subfolders (e.g. a diffusers repo's `unet/`, `vae/`).
fn sum_model_weight_bytes(dir: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let Ok(children) = std::fs::read_dir(dir) else { return total; };
    for child in children.flatten() {
        let path = child.path();
        if path.is_dir() {
            total += sum_model_weight_bytes(&path);
        } else if path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "safetensors" | "gguf" | "bin" | "pth" | "ckpt")) {
            total += child.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        }
    }
    total
}

/// If `dir` is directly a HuggingFace model directory — it contains its own
/// `config.json` (transformers-style) or `model_index.json` (diffusers-style)
/// — build one collapsed `DetectedModelEntry` for the whole directory.
/// Returns `None` for plain directories, so the caller keeps recursing.
fn hf_model_dir_entry(dir: &std::path::Path) -> Option<DetectedModelEntry> {
    let has_config = dir.join("config.json").is_file();
    let has_model_index = dir.join("model_index.json").is_file();
    if !has_config && !has_model_index {
        return None;
    }

    let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_ascii_lowercase();
    let is_mesh3d = is_mesh3d_name(&dir_name);
    let task_tags = if let Some(hint) = read_task_tags_hint(dir) {
        hint
    } else if is_mesh3d {
        crate::task_tags::mesh_tags_from_kinds(detect_mesh_input_kinds(dir))
    } else if has_model_index {
        vec!["text-to-image".to_string()]
    } else {
        crate::task_tags::local_heuristic_tags_for_dir(dir)
    };
    let is_mesh3d = is_mesh3d || task_tags.iter().any(|t| t == "text-to-3d" || t == "image-to-3d");
    let mesh_input_kinds = if is_mesh3d { detect_mesh_input_kinds(dir) } else { None };
    let (mesh_vae_path, mesh_texgen_path) = if is_mesh3d { find_mesh3d_siblings(dir) } else { (None, None) };

    Some(DetectedModelEntry {
        name: dir.file_name().and_then(|name| name.to_str()).unwrap_or("Unknown model").to_string(),
        path: dir.to_string_lossy().to_string(),
        size_bytes: sum_model_weight_bytes(dir),
        max_context_size: None,
        image_input_available: crate::task_tags::image_input_available(&task_tags),
        mmproj_path: None,
        mesh_input_kinds,
        mesh_vae_path,
        mesh_texgen_path,
        image_components: None,
        task_tags,
    })
}

fn collect_model_files(directory: &std::path::Path, entries: &mut Vec<DetectedModelEntry>) {
    let Ok(children) = std::fs::read_dir(directory) else { return; };
    for child in children.flatten() {
        let path = child.path();
        if path.is_dir() {
            // Multi-component 3D mesh repos (e.g. Hunyuan3D-2.1) have no root
            // config.json / model_index.json, so hf_model_dir_entry() would miss
            // them and the scanner would recurse into subfolders (dit/vae/...),
            // emitting nested weight files as standalone models. The mesh3d
            // backend needs the repo-root folder to locate those subfolders, so
            // short-circuit on the dir name before recursing.
            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_ascii_lowercase();
            // Skip "shared" folder — it contains cross-backend component files
            // (shared VAE, text-encoders, etc.), not standalone models.
            if dir_name == "shared" {
                continue;
            }
            if is_mesh3d_name(&dir_name) {
                let (mesh_vae_path, mesh_texgen_path) = find_mesh3d_siblings(&path);
                let mesh_input_kinds = detect_mesh_input_kinds(&path);
                entries.push(DetectedModelEntry {
                    name: path.file_name().and_then(|name| name.to_str()).unwrap_or("Unknown model").to_string(),
                    path: path.to_string_lossy().to_string(),
                    size_bytes: sum_model_weight_bytes(&path),
                    max_context_size: None,
                    image_input_available: false,
                    mmproj_path: None,
                    task_tags: crate::task_tags::mesh_tags_from_kinds(mesh_input_kinds.clone()),
                    mesh_input_kinds,
                    mesh_vae_path,
                    mesh_texgen_path,
                    image_components: None,
                });
                continue;
            }
            if let Some(entry) = hf_model_dir_entry(&path) {
                // A HF model dir is emitted as ONE entry — don't also descend
                // into it to emit per-file entries for its shards/components.
                entries.push(entry);
                continue;
            }
            collect_model_files(&path, entries);
        } else if path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "gguf" | "safetensors" | "ckpt" | "onnx" | "pth" | "pt" | "bin" | "ot" | "tflite" | "mlmodel" | "engine")) {
            // Skip mmproj companion files — they are not standalone models
            let file_name_lower = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
            if file_name_lower.contains("mmproj") {
                continue;
            }
            let size_bytes = child.metadata().map(|metadata| metadata.len()).unwrap_or(0);
            let task_tags = path
                .parent()
                .and_then(read_task_tags_hint)
                .unwrap_or_else(|| crate::task_tags::local_heuristic_tags_for_file(&path));
            let is_mesh3d = task_tags.iter().any(|t| t == "text-to-3d" || t == "image-to-3d");
            let mesh_input_kinds = if is_mesh3d { detect_mesh_input_kinds(&path) } else { None };
            let sibling_mmproj = find_sibling_mmproj(&path).map(|p| p.to_string_lossy().to_string());
            let is_image = task_tags.iter().any(|t| t == "text-to-image" || t == "image-to-image");
            let image_components = if is_image { sd_backend::SdBackend::detect_image_components(&path) } else { None };
            entries.push(DetectedModelEntry {
                name: path.file_name().and_then(|name| name.to_str()).unwrap_or("Unknown model").to_string(),
                path: path.to_string_lossy().to_string(),
                size_bytes,
                max_context_size: gguf_context_length(&path),
                image_input_available: crate::task_tags::image_input_available(&task_tags) || sibling_mmproj.is_some(),
                mmproj_path: sibling_mmproj,
                mesh_input_kinds,
                mesh_vae_path: None,
                mesh_texgen_path: None,
                image_components,
                task_tags,
            });
        }
    }
}

/// Locate DisposAI's own `models` directory without depending on how the daemon
/// was launched. This supports development builds, packaged binaries, and an
/// explicit deployment override through `DISPOS_MODELS_DIR`.
/// Resolve the local models directory using the same search order as the
/// model catalog. Returns the first existing directory found.
fn resolve_models_dir() -> Option<std::path::PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(directory) = std::env::var("DISPOS_MODELS_DIR") {
        candidates.push(std::path::PathBuf::from(directory));
    }
    if let Ok(current_dir) = std::env::current_dir() {
        candidates.extend(current_dir.ancestors().map(|path| path.join("models")));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.extend(parent.ancestors().map(|path| path.join("models")));
        }
    }
    // Cargo resolves this to the project source tree, which keeps local
    // development reliable even when the binary is built into another target dir.
    candidates.push(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..\\..").join("models"));

    candidates.into_iter().find(|directory| directory.is_dir())
}

pub(crate) async fn list_detected_models() -> Json<Vec<DetectedModelEntry>> {
    let mut entries = Vec::new();

    if let Some(directory) = resolve_models_dir() {
        collect_model_files(&directory, &mut entries);
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    for e in &entries {
        tracing::info!("catalog entry: name={}, path={}, mmproj_path={:?}", e.name, e.path, e.mmproj_path);
    }
    Json(entries)
}

/// Returns the backwards-compat single model status (first loaded model, or none).
async fn get_model_status(
    State(state): State<AppState>,
) -> Json<LoadedModelStatus> {
    let models = state.loaded_models.lock().await;
    if let Some(entry) = models.values().next() {
        Json(LoadedModelStatus {
            is_loaded: true,
            model_path: Some(entry.model_path.clone()),
            gpu_layers: Some(entry.gpu_layers),
            context_size: Some(entry.context_size),
        })
    } else {
        Json(LoadedModelStatus { is_loaded: false, model_path: None, gpu_layers: None, context_size: None })
    }
}

/// Returns all currently loaded models.
async fn list_loaded_models(
    State(state): State<AppState>,
) -> Json<Vec<LoadedModelEntry>> {
    let models = state.loaded_models.lock().await;
    // IDs contain a creation timestamp, so this makes the API's default model
    // choice deterministic: the most recently loaded model appears first.
    let mut entries: Vec<_> = models.values().cloned().collect();
    entries.sort_by(|a, b| b.model_id.cmp(&a.model_id));
    Json(entries)
}

// ---------------------------------------------------------------------------
// Hugging Face model search & download
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct HfSearchQuery {
    q: Option<String>,
    sort: Option<String>,
    filter: Option<String>,
    min_params: Option<u64>,
    max_params: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfExpandTotal {
    total: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct HfModelResult {
    id: String,
    author: Option<String>,
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    #[serde(default)]
    downloads: i64,
    #[serde(default)]
    likes: i64,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
    #[serde(default)]
    private: bool,
    #[serde(skip_deserializing)]
    params: Option<u64>,
    #[serde(default, skip_serializing)]
    safetensors: Option<HfExpandTotal>,
    #[serde(default, skip_serializing)]
    gguf: Option<HfExpandTotal>,
}

/// The on-disk model formats this daemon can actually run, without dots, lowercase.
/// Single source of truth for catalog gating.
const SUPPORTED_FORMATS: &[&str] = &["gguf", "safetensors"];

/// Sanitize HF token by keeping only ASCII graphic characters (0x21-0x7E).
/// This strips all invisible Unicode characters that may have been pasted from browsers.
fn sanitize_hf_token(raw: &str) -> String {
    raw.chars().filter(|c| c.is_ascii_graphic()).collect()
}

/// Read HF token: runtime override (from UI) > HF_TOKEN env var > cached token file.
async fn resolve_hf_token(state_token: &Arc<tokio::sync::Mutex<Option<String>>>) -> Option<String> {
    {
        let lock = state_token.lock().await;
        if let Some(ref t) = *lock {
            if !t.is_empty() {
                return Some(t.clone());
            }
        }
    }
    if let Ok(t) = std::env::var("HF_TOKEN") {
        let t = sanitize_hf_token(&t);
        if !t.is_empty() {
            return Some(t);
        }
    }
    if let Ok(t) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        let t = sanitize_hf_token(&t);
        if !t.is_empty() {
            return Some(t);
        }
    }
    if let Ok(t) = tokio::fs::read_to_string(hf_token_cache_path()?).await {
        let t = sanitize_hf_token(&t);
        if !t.is_empty() {
            return Some(t);
        }
    }
    None
}

/// Path to the standard Hugging Face CLI token cache file, used both as a
/// read fallback and as where the UI-set token is persisted across restarts.
fn hf_token_cache_path() -> Option<std::path::PathBuf> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
    Some(std::path::PathBuf::from(home).join(".cache").join("huggingface").join("token"))
}

async fn hf_search(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HfSearchQuery>,
) -> Json<Vec<HfModelResult>> {
    // NOTE: the Hugging Face `filter` param ANDs its tags together, so a single
    // request seeded with every entry in SUPPORTED_FORMATS would wrongly require
    // a model to match ALL formats at once (e.g. "gguf AND safetensors" -> no
    // results). Instead issue one request per supported format (OR semantics)
    // and merge, deduping by model id and preserving order.
    let mut format_tags: Vec<&str> = Vec::new();
    let mut extra_tags: Vec<String> = Vec::new();
    if let Some(filter) = &params.filter {
        for tag in filter.split(',') {
            let tag = tag.trim();
            if tag.is_empty() {
                continue;
            }
            if let Some(&format) = SUPPORTED_FORMATS.iter().find(|f| **f == tag) {
                if !format_tags.contains(&format) {
                    format_tags.push(format);
                }
            } else {
                extra_tags.push(tag.to_string());
            }
        }
    }
    if format_tags.is_empty() && extra_tags.is_empty() {
        format_tags = SUPPORTED_FORMATS.to_vec();
    }
    let sort = params.sort.unwrap_or_else(|| "trendingScore".to_string());
    // With no search term the panel is in "browse" mode, so surface a fuller top
    // list; a real query narrows things down and 20 keeps it snappy.
    let has_query = params.q.as_ref().is_some_and(|q| !q.is_empty());
    let limit = if has_query { 20usize } else { 50usize };

    let client = match reqwest::Client::builder()
        .user_agent("dispos-ai/0.1")
        .build()
    {
        Ok(c) => c,
        Err(_) => return Json(Vec::new()),
    };

    let token = resolve_hf_token(&state.hf_token).await;

    let mut merged: Vec<HfModelResult> = Vec::new();
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    // When no format tags, do a single pass with just the extra tags.
    // When format tags exist, do one pass per format (OR semantics).
    let passes: Vec<&str> = if format_tags.is_empty() { vec![""] } else { format_tags };
    for format in &passes {
        if merged.len() >= limit {
            break;
        }
        let mut filter_tags: Vec<String> = if format.is_empty() {
            Vec::new()
        } else {
            vec![format.to_string()]
        };
        filter_tags.extend(extra_tags.iter().cloned());

        let mut request = client
            .get("https://huggingface.co/api/models")
            .query(&[("limit", limit.to_string().as_str()), ("sort", sort.as_str())])
            .query(&[("filter", filter_tags.join(","))])
            // Once ANY expand[] is present the HF API returns *only* the id plus
            // the fields explicitly expanded, so every field HfModelResult reads
            // must be listed here or it silently deserializes to its default
            // (which is why downloads/likes were always 0).
            .query(&[
                ("expand[]", "safetensors"),
                ("expand[]", "gguf"),
                ("expand[]", "downloads"),
                ("expand[]", "likes"),
                ("expand[]", "tags"),
                ("expand[]", "author"),
                ("expand[]", "private"),
                ("expand[]", "lastModified"),
            ]);

        if let Some(q) = params.q.as_ref().filter(|q| !q.is_empty()) {
            request = request.query(&[("search", q.as_str())]);
        }

        if let Some(ref t) = token {
            request = request.header("Authorization", format!("Bearer {}", t));
        }

        let response = match request.send().await {
            Ok(resp) => resp,
            Err(err) => {
                tracing::warn!("hf-search request failed for format '{format}': {err}");
                continue;
            }
        };

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            tracing::warn!("hf-search returned error for format '{format}': {body}");
            continue;
        }

        match response.json::<Vec<HfModelResult>>().await {
            Ok(results) => {
                for mut result in results {
                    if merged.len() >= limit {
                        break;
                    }
                    result.params = result
                        .safetensors
                        .as_ref()
                        .and_then(|s| s.total)
                        .or_else(|| result.gguf.as_ref().and_then(|g| g.total));
                    if seen_ids.insert(result.id.clone()) {
                        merged.push(result);
                    }
                }
            }
            Err(err) => {
                tracing::warn!("hf-search parse failed for format '{format}': {err}");
            }
        }
    }

    if params.min_params.is_some() || params.max_params.is_some() {
        merged.retain(|result| match result.params {
            Some(p) => {
                params.min_params.map_or(true, |m| p >= m)
                    && params.max_params.map_or(true, |m| p <= m)
            }
            None => false,
        });
    }

    Json(merged)
}

#[derive(Debug, Deserialize)]
struct HfFilesQuery {
    repo: String,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    rfilename: String,
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfRepoInfo {
    #[serde(default)]
    siblings: Vec<HfSibling>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

/// Filename of the per-model-directory sidecar written after download that
/// caches the HF `pipeline_tag`-derived task tag(s), so a later local scan
/// doesn't have to re-derive them from directory contents alone. Stores a
/// JSON array of tags. An old plain-text `.dispos-modality` sidecar (from
/// before this format existed) simply fails to parse as JSON and is ignored —
/// it self-heals the next time `write_task_tags_hint` runs after a fresh
/// HF fetch.
const TASK_TAGS_HINT_FILE: &str = ".dispos-task-tags";

/// Read a cached task-tag hint written by `write_task_tags_hint`, if present
/// and valid. Each tag is validated against `task_tags::known_task_tag`;
/// unrecognized entries are dropped. Returns `None` if the file is missing,
/// isn't valid JSON, or has no recognized tags left after filtering, so the
/// caller falls back to the local heuristic.
fn read_task_tags_hint(dir: &std::path::Path) -> Option<Vec<String>> {
    let contents = std::fs::read_to_string(dir.join(TASK_TAGS_HINT_FILE)).ok()?;
    let raw: Vec<String> = serde_json::from_str(&contents).ok()?;
    let valid: Vec<String> = raw
        .into_iter()
        .filter(|tag| crate::task_tags::known_task_tag(tag).is_some())
        .collect();
    (!valid.is_empty()).then_some(valid)
}

/// Best-effort: fetch the repo's `pipeline_tag` from the HF API and cache the
/// derived task tag as a sidecar file in its model directory. Failures are
/// silently ignored — the local heuristic remains the fallback.
async fn write_task_tags_hint(repo: &str, dir: &std::path::Path, hf_token: Option<String>) {
    let Ok(client) = reqwest::Client::builder().user_agent("dispos-ai/0.1").build() else { return };
    let mut req = client.get(format!("https://huggingface.co/api/models/{repo}"));
    if let Some(t) = hf_token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let Ok(response) = req.send().await else { return };
    let Ok(info) = response.json::<HfRepoInfo>().await else { return };
    let mut tags: Vec<&str> = Vec::new();
    if let Some(tag) = info.pipeline_tag.as_deref().and_then(crate::task_tags::known_task_tag) {
        tags.push(tag);
    }
    for t in info.tags.iter().filter_map(|t| crate::task_tags::known_task_tag(t)) {
        if !tags.contains(&t) {
            tags.push(t);
        }
    }
    if tags.is_empty() {
        return;
    }
    let Ok(json) = serde_json::to_string(&tags) else { return };
    let _ = tokio::fs::write(dir.join(TASK_TAGS_HINT_FILE), json).await;
}

#[derive(Debug, Deserialize)]
struct HfTreeLfs {
    size: u64,
}

#[derive(Debug, Deserialize)]
struct HfTreeEntry {
    path: String,
    #[serde(default, rename = "type")]
    r#type: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<HfTreeLfs>,
}

#[derive(Debug, Serialize)]
struct HfFilesResponse {
    files: Vec<HfFileEntry>,
    kind: String,
    autodownload: Vec<String>,
    autodownload_reason: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct HfFileEntry {
    filename: String,
    size: Option<u64>,
    role: String,
    quant: Option<String>,
    variant_group: Option<String>,
}

fn hf_file_base_name(rfilename: &str) -> &str {
    rfilename.rsplit('/').next().unwrap_or(rfilename)
}

/// The parent directory of a repo-relative filename, or `""` for root-level
/// files. Used to scope weight-file variant grouping to files that live
/// side by side in the same folder.
fn parent_dir(filename: &str) -> &str {
    filename.rfind('/').map(|i| &filename[..i]).unwrap_or("")
}

/// `-\d+-of-\d+\.(safetensors|gguf)` — `base` must already be lowercased.
fn is_shard_filename(base: &str) -> bool {
    let stem = if let Some(stem) = base.strip_suffix(".safetensors") {
        stem
    } else if let Some(stem) = base.strip_suffix(".gguf") {
        stem
    } else {
        return false;
    };
    let Some(of_idx) = stem.rfind("-of-") else { return false };
    let after = &stem[of_idx + 4..];
    if after.is_empty() || !after.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let before = &stem[..of_idx];
    let Some(dash_idx) = before.rfind('-') else { return false };
    let num_part = &before[dash_idx + 1..];
    !num_part.is_empty() && num_part.chars().all(|c| c.is_ascii_digit())
}

/// Classify a repo file's role from its (repo-relative) path, in the order:
/// index → shard → mmproj → config → tokenizer → unet/vae/text_encoder →
/// standalone weights → other.
fn classify_hf_file_role(rfilename: &str) -> String {
    let lower = rfilename.to_ascii_lowercase();
    let base = hf_file_base_name(&lower);

    if base.ends_with(".index.json") {
        return "index".to_string();
    }
    if is_shard_filename(base) {
        return "shard".to_string();
    }
    if base.contains("mmproj") {
        return "mmproj".to_string();
    }
    if base == "config.json" || base == "generation_config.json" || base == "config.yaml" {
        return "config".to_string();
    }
    if base.starts_with("tokenizer") || matches!(base, "vocab.json" | "merges.txt" | "special_tokens_map.json") {
        return "tokenizer".to_string();
    }
    if lower.split('/').any(|segment| segment == "unet") {
        return "unet".to_string();
    }
    if lower.split('/').any(|segment| segment == "vae") {
        return "vae".to_string();
    }
    if lower.split('/').any(|segment| segment == "text_encoder" || segment == "text_encoder_2") {
        return "text_encoder".to_string();
    }
    if base.ends_with(".safetensors")
        || base.ends_with(".gguf")
        || base.ends_with(".ckpt")
        || base.ends_with(".bin")
        || base.ends_with(".pt")
        || base.ends_with(".pth")
    {
        return "weights".to_string();
    }
    "other".to_string()
}

/// Extract a canonical quant/precision token (e.g. "Q4_K_M", "Q8_0", "F16",
/// "BF16") from a filename. Case-insensitive; returns the uppercased token.
fn extract_quant_token(filename: &str) -> Option<String> {
    let upper = filename.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'Q' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            let boundary_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            if !boundary_ok {
                continue;
            }
            let mut end = i + 1;
            while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
                end += 1;
            }
            while end > i + 1 && bytes[end - 1] == b'_' {
                end -= 1;
            }
            return Some(upper[i..end].to_string());
        }
    }
    for pattern in ["BF16", "FP16", "FP8", "INT8", "F16", "F32"] {
        if let Some(idx) = upper.find(pattern) {
            let before_ok = idx == 0 || !bytes[idx - 1].is_ascii_alphanumeric();
            let after = idx + pattern.len();
            let after_ok = after >= bytes.len() || !bytes[after].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return Some(pattern.to_string());
            }
        }
    }
    None
}

/// The weight-file format of a standalone (non-shard) weights filename, used
/// to group interchangeable quant/precision variants.
fn standalone_weight_ext(filename: &str) -> Option<&'static str> {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".safetensors") {
        Some("safetensors")
    } else if lower.ends_with(".gguf") {
        Some("gguf")
    } else if lower.ends_with(".ckpt") {
        Some("ckpt")
    } else if lower.ends_with(".bin") {
        Some("bin")
    } else if lower.ends_with(".pt") {
        Some("pt")
    } else if lower.ends_with(".pth") {
        Some("pth")
    } else {
        None
    }
}

/// Pick the best-fit weight file among a "variants" repo's standalone weight
/// alternatives, sized against `free_vram_bytes`. Returns the chosen entry
/// plus a human-readable reason.
fn pick_variant_weights_file<'a>(
    entries: &'a [HfFileEntry],
    free_vram_bytes: u64,
) -> Option<(&'a HfFileEntry, String)> {
    let variants: Vec<&HfFileEntry> = entries.iter().filter(|e| e.role == "weights" && e.variant_group.is_some()).collect();
    pick_best_variant(&variants, free_vram_bytes)
}

/// Core "pick the best-fitting weight file among a set of interchangeable
/// variants" logic, sized against `free_vram_bytes`: a variant with a known
/// free-VRAM fit else the largest that fits else the smallest overall.
/// Shared by the top-level "variants" repo kind and per-component grouping in
/// "components" repos.
fn pick_best_variant<'a>(variants: &[&'a HfFileEntry], free_vram_bytes: u64) -> Option<(&'a HfFileEntry, String)> {
    if variants.is_empty() {
        return None;
    }
    let free_gb = free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let sized: Vec<&HfFileEntry> = variants.iter().copied().filter(|e| e.size.is_some()).collect();

    if sized.is_empty() {
        let chosen = variants.iter().copied().find(|e| e.quant.as_deref() == Some("Q4_K_M")).unwrap_or(variants[0]);
        let label = chosen.quant.clone().unwrap_or_else(|| chosen.filename.clone());
        return Some((chosen, format!("picked {} (size unknown)", label)));
    }

    let fitting: Vec<&HfFileEntry> = sized
        .iter()
        .copied()
        .filter(|e| (e.size.unwrap() as f64) * 1.2 <= free_vram_bytes as f64)
        .collect();

    if let Some(best) = fitting.iter().copied().max_by_key(|e| e.size.unwrap()) {
        let size_gb = best.size.unwrap() as f64 / (1024.0 * 1024.0 * 1024.0);
        let label = best.quant.clone().unwrap_or_else(|| best.filename.clone());
        return Some((best, format!("picked {} ({:.1} GB) — fits {:.1} GB free VRAM", label, size_gb, free_gb)));
    }

    let smallest = sized.iter().copied().min_by_key(|e| e.size.unwrap())?;
    let label = smallest.quant.clone().unwrap_or_else(|| smallest.filename.clone());
    Some((smallest, format!("picked {} (smallest) — may exceed {:.1} GB free VRAM", label, free_gb)))
}

/// Extracts the `rel="next"` URL from an RFC 5988 `Link` header value,
/// as returned by HF's paginated tree API.
fn parse_hf_link_next(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        let part = part.trim();
        if !part.contains("rel=\"next\"") {
            return None;
        }
        let start = part.find('<')?;
        let end = part.find('>')?;
        if end > start {
            Some(part[start + 1..end].to_string())
        } else {
            None
        }
    })
}

async fn hf_files(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HfFilesQuery>,
) -> Json<HfFilesResponse> {
    fn empty_response() -> HfFilesResponse {
        HfFilesResponse { files: Vec::new(), kind: "single".to_string(), autodownload: Vec::new(), autodownload_reason: None }
    }

    let client = match reqwest::Client::builder()
        .user_agent("dispos-ai/0.1")
        .build()
    {
        Ok(c) => c,
        Err(_) => return Json(empty_response()),
    };

    let token = resolve_hf_token(&state.hf_token).await;

    let url = format!("https://huggingface.co/api/models/{}", params.repo);
    let mut req = client.get(&url);
    if let Some(ref t) = token {
        req = req.header("Authorization", format!("Bearer {}", t));
    }
    let response = match req.send().await {
        Ok(resp) => resp,
        Err(err) => {
            tracing::warn!("hf-files request failed: {err}");
            return Json(empty_response());
        }
    };

    let info = match response.json::<HfRepoInfo>().await {
        Ok(info) => info,
        Err(err) => {
            tracing::warn!("hf-files parse failed: {err}");
            return Json(empty_response());
        }
    };

    // The recursive tree API is the authoritative complete file list for the
    // repo (siblings entries can omit subfolder contents' full detail and
    // leave `size` = None for large LFS files). Fetch it and build the file
    // list from `type == "file"` entries, using `lfs.size` else `size` for
    // each file's size. Fall back to `info.siblings` if the tree fetch fails
    // or yields no file entries.
    // The HF tree API paginates large repos via an RFC 5988 `Link: <url>; rel="next"`
    // response header rather than a body cursor. Repos with many files/subfolders
    // (e.g. MiniMaxAI/MiniMax-H3) span multiple pages, so we must follow `next`
    // links until exhausted or a request fails, otherwise later pages are silently
    // dropped and the catalog looks truncated.
    let mut tree_files: Vec<(String, Option<u64>)> = Vec::new();
    let mut next_url = Some(format!(
        "https://huggingface.co/api/models/{}/tree/main?recursive=true&expand=1",
        params.repo
    ));
    const MAX_TREE_PAGES: usize = 100;
    for _ in 0..MAX_TREE_PAGES {
        let Some(url) = next_url.take() else { break };
        let mut tree_req = client.get(&url);
        if let Some(ref t) = token {
            tree_req = tree_req.header("Authorization", format!("Bearer {}", t));
        }
        match tree_req.send().await {
            Ok(resp) => {
                next_url = resp
                    .headers()
                    .get("link")
                    .and_then(|v| v.to_str().ok())
                    .and_then(parse_hf_link_next);
                match resp.json::<Vec<HfTreeEntry>>().await {
                    Ok(entries) => {
                        for entry in entries {
                            if entry.r#type != "file" {
                                continue;
                            }
                            let size = entry.lfs.map(|lfs| lfs.size).or(entry.size);
                            tree_files.push((entry.path, size));
                        }
                    }
                    Err(err) => {
                        tracing::warn!("hf-files tree parse failed: {err}");
                        break;
                    }
                }
            }
            Err(err) => {
                tracing::warn!("hf-files tree request failed: {err}");
                break;
            }
        }
    }

    let mut files: Vec<HfFileEntry> = if !tree_files.is_empty() {
        tree_files
            .into_iter()
            .map(|(filename, size)| {
                let role = classify_hf_file_role(&filename);
                let quant = extract_quant_token(hf_file_base_name(&filename));
                HfFileEntry { filename, size, role, quant, variant_group: None }
            })
            .collect()
    } else {
        info.siblings
            .into_iter()
            .map(|s| {
                let role = classify_hf_file_role(&s.rfilename);
                let quant = extract_quant_token(hf_file_base_name(&s.rfilename));
                HfFileEntry { filename: s.rfilename, size: s.size, role, quant, variant_group: None }
            })
            .collect()
    };

    // Group standalone weight files into interchangeable variants when a repo
    // offers 2+ alternatives of the same weight format IN THE SAME DIRECTORY
    // (different quants/precisions of the same model living side by side).
    // Directory-scoped so a component-pipeline repo's per-subfolder weights
    // (e.g. transformer/model.safetensors) never get lumped in with an
    // unrelated root-level convenience copy of the same extension.
    let mut variant_counts: std::collections::HashMap<(String, &'static str), usize> = std::collections::HashMap::new();
    for f in files.iter() {
        if !matches!(f.role.as_str(), "weights" | "unet" | "vae" | "text_encoder") {
            continue;
        }
        if let Some(ext) = standalone_weight_ext(&f.filename) {
            *variant_counts.entry((parent_dir(&f.filename).to_string(), ext)).or_insert(0) += 1;
        }
    }
    for f in files.iter_mut() {
        if !matches!(f.role.as_str(), "weights" | "unet" | "vae" | "text_encoder") {
            continue;
        }
        if let Some(ext) = standalone_weight_ext(&f.filename) {
            let dir = parent_dir(&f.filename).to_string();
            if variant_counts.get(&(dir, ext)).copied().unwrap_or(0) >= 2 {
                f.variant_group = Some(ext.to_string());
            }
        }
    }

    let has_components = files.iter().any(|f| matches!(f.role.as_str(), "unet" | "vae" | "text_encoder"))
        || files.iter().any(|f| hf_file_base_name(&f.filename) == "model_index.json");
    let has_shard_or_index = files.iter().any(|f| matches!(f.role.as_str(), "shard" | "index"));
    let standalone_weight_count = files.iter().filter(|f| f.role == "weights").count();

    let kind = if has_components {
        "components"
    } else if has_shard_or_index {
        "sharded"
    } else if standalone_weight_count >= 2 {
        "variants"
    } else {
        "single"
    }
    .to_string();

    let mmproj_filename = files.iter().find(|f| f.role == "mmproj").map(|f| f.filename.clone());

    let (autodownload, autodownload_reason): (Vec<String>, Option<String>) = match kind.as_str() {
        "sharded" => {
            let shard_count = files.iter().filter(|f| f.role == "shard").count();
            let mut names: Vec<String> = files
                .iter()
                .filter(|f| matches!(f.role.as_str(), "shard" | "index" | "config" | "tokenizer"))
                .map(|f| f.filename.clone())
                .collect();
            if let Some(mmproj) = &mmproj_filename {
                names.push(mmproj.clone());
            }
            (names, Some(format!("Complete {}-shard set + config & tokenizer", shard_count)))
        }
        "components" => {
            let sys = HardwareProfiler::probe();
            let free_vram_bytes = if sys.free_vram_bytes > 0 { sys.free_vram_bytes } else { sys.available_ram_bytes };

            let mut groups: std::collections::HashMap<(String, String), Vec<&HfFileEntry>> = std::collections::HashMap::new();
            for f in files.iter() {
                if let Some(group) = &f.variant_group {
                    groups.entry((parent_dir(&f.filename).to_string(), group.clone())).or_default().push(f);
                }
            }

            let mut names: Vec<String> = files
                .iter()
                .filter(|f| f.variant_group.is_none())
                .map(|f| f.filename.clone())
                .collect();
            for variants in groups.values() {
                if let Some((chosen, _reason)) = pick_best_variant(variants, free_vram_bytes) {
                    names.push(chosen.filename.clone());
                }
            }
            (names, Some("All pipeline components (best-fit variant per part)".to_string()))
        }
        "variants" => {
            let sys = HardwareProfiler::probe();
            let free_vram_bytes = if sys.free_vram_bytes > 0 { sys.free_vram_bytes } else { sys.available_ram_bytes };
            match pick_variant_weights_file(&files, free_vram_bytes) {
                Some((chosen, reason)) => {
                    let mut names = vec![chosen.filename.clone()];
                    names.extend(
                        files
                            .iter()
                            .filter(|f| matches!(f.role.as_str(), "config" | "tokenizer"))
                            .map(|f| f.filename.clone()),
                    );
                    if let Some(mmproj) = &mmproj_filename {
                        names.push(mmproj.clone());
                    }
                    (names, Some(reason))
                }
                None => (Vec::new(), None),
            }
        }
        _ => {
            // "single"
            let mut names: Vec<String> = files
                .iter()
                .filter(|f| matches!(f.role.as_str(), "weights" | "config" | "tokenizer"))
                .map(|f| f.filename.clone())
                .collect();
            if let Some(mmproj) = &mmproj_filename {
                names.push(mmproj.clone());
            }
            (names, Some("Model weights + config".to_string()))
        }
    };

    Json(HfFilesResponse { files, kind, autodownload, autodownload_reason })
}

#[derive(Debug, Deserialize)]
struct HfDownloadRequest {
    repo: String,
    filename: String,
    /// Optional destination subfolder relative to the models root (e.g.
    /// "shared/vae"). When set alongside `target_filename`, overrides the
    /// default `<models_dir>/<sanitized repo id>/<filename>` layout — used
    /// to drop split-checkpoint components (VAE, text encoders) straight
    /// into sd-backend's shared components folders.
    #[serde(default)]
    target_dir: Option<String>,
    /// Optional destination filename, used together with `target_dir`. Lets
    /// an upstream file (e.g. BFL's `ae.safetensors`) be saved under a
    /// renamed filename that satisfies `ComponentKind::filename_patterns()`.
    #[serde(default)]
    target_filename: Option<String>,
}

#[derive(Debug, Serialize)]
struct HfDownloadResponse {
    status: &'static str,
    path: String,
}

async fn hf_download(
    State(state): State<AppState>,
    Json(payload): Json<HfDownloadRequest>,
) -> Json<HfDownloadResponse> {
    let models_dir = resolve_models_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("models"));
    // Save each repo's files into its own subfolder so config.json/tokenizer.json
    // from different models don't collide in the shared models/ root — unless
    // the caller explicitly asked for a specific destination (e.g. dropping a
    // split-checkpoint component into models/shared/vae).
    let save_path = match (&payload.target_dir, &payload.target_filename) {
        (Some(dir), Some(name)) => models_dir.join(dir).join(name),
        _ => models_dir
            .join(sanitize_repo_id(&payload.repo))
            .join(&payload.filename),
    };
    let save_path_string = save_path.to_string_lossy().to_string();

    let key = download_key(&payload.repo, &payload.filename);
    {
        let mut downloads = state.hf_downloads.lock().await;
        downloads.insert(
            key.clone(),
            DownloadProgress {
                filename: payload.filename.clone(),
                repo: payload.repo.clone(),
                total_bytes: None,
                downloaded_bytes: 0,
                status: "downloading".to_string(),
                error: None,
            },
        );
    }

    let cancel_flag = Arc::new(AtomicBool::new(false));
    {
        let mut cancel_map = state.hf_cancel.lock().await;
        cancel_map.insert(key.clone(), cancel_flag.clone());
    }

    let downloads_map = state.hf_downloads.clone();
    let repo = payload.repo.clone();
    let filename = payload.filename.clone();
    let target_path = save_path.clone();
    let hf_token = resolve_hf_token(&state.hf_token).await;

    // Only the default per-repo layout (no target_dir override) produces a
    // model directory worth tagging — shared components (VAE, text encoders)
    // land in a shared folder that isn't itself a model.
    if payload.target_dir.is_none() {
        if let Some(model_dir) = save_path.parent().map(|p| p.to_path_buf()) {
            let hint_repo = payload.repo.clone();
            let hint_token = hf_token.clone();
            tokio::spawn(async move {
                write_task_tags_hint(&hint_repo, &model_dir, hint_token).await;
            });
        }
    }

    tokio::spawn(async move {
        if let Err(err) = run_hf_download(&repo, &filename, &target_path, &downloads_map, &cancel_flag, hf_token).await {
            tracing::warn!("hf-download failed for {filename}: {err}");
            let mut downloads = downloads_map.lock().await;
            if let Some(entry) = downloads.get_mut(&key) {
                entry.status = "error".to_string();
                entry.error = Some(err.to_string());
            }
        }
    });

    Json(HfDownloadResponse {
        status: "downloading",
        path: save_path_string,
    })
}

#[derive(Debug, Deserialize)]
struct HfTokenRequest {
    token: String,
}

#[derive(Debug, Serialize)]
struct HfTokenResponse {
    status: &'static str,
    has_token: bool,
}

async fn hf_set_token(
    State(state): State<AppState>,
    Json(payload): Json<HfTokenRequest>,
) -> Json<HfTokenResponse> {
    let mut lock = state.hf_token.lock().await;
    let sanitized = sanitize_hf_token(&payload.token);
    let has = !sanitized.is_empty();
    if let Some(path) = hf_token_cache_path() {
        if has {
            if let Some(parent) = path.parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            let _ = tokio::fs::write(&path, &sanitized).await;
        } else {
            let _ = tokio::fs::remove_file(&path).await;
        }
    }
    *lock = if has { Some(sanitized) } else { None };
    Json(HfTokenResponse { status: "ok", has_token: has })
}

async fn hf_get_token_status(
    State(state): State<AppState>,
) -> Json<HfTokenResponse> {
    let token = resolve_hf_token(&state.hf_token).await;
    Json(HfTokenResponse { status: "ok", has_token: token.is_some() })
}

#[derive(Debug, Deserialize)]
struct ModelPathRequest {
    path: String,
    /// Optional companion vision projector file belonging exclusively to this
    /// model — deleted alongside it if present. Never used to widen deletion
    /// to anything else in the folder.
    #[serde(default)]
    mmproj_path: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelActionResponse {
    status: &'static str,
}

async fn open_model_folder(
    Json(payload): Json<ModelPathRequest>,
) -> Result<Json<ModelActionResponse>, (axum::http::StatusCode, String)> {
    let path = std::path::Path::new(&payload.path);
    if !path.exists() {
        return Err((axum::http::StatusCode::NOT_FOUND, "Model path not found".into()));
    }
    let parent = path.parent()
        .ok_or((axum::http::StatusCode::BAD_REQUEST, "Model path has no parent directory".into()))?;

    std::process::Command::new("explorer")
        .arg(parent)
        .spawn()
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(ModelActionResponse { status: "ok" }))
}

async fn delete_model(
    Json(payload): Json<ModelPathRequest>,
) -> Result<Json<ModelActionResponse>, (axum::http::StatusCode, String)> {
    let path = std::path::Path::new(&payload.path);
    if !path.exists() {
        return Err((axum::http::StatusCode::NOT_FOUND, "Model path not found".into()));
    }

    let models_root = resolve_models_dir()
        .ok_or((axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Could not resolve models directory".into()))?;
    let models_root_canon = models_root.canonicalize().unwrap_or_else(|_| models_root.clone());

    // A catalog entry's `path` either points directly at a model's own
    // exclusive directory (HF-style repos, mesh3d bundles — safe to remove
    // wholesale) or at a single loose file living alongside other models'
    // files (e.g. GGUFs dropped straight in the models folder or sharing a
    // repo subfolder). For the file case we only ever remove that one file
    // (plus its own mmproj, if given) — never its containing folder — so a
    // single delete can never take out sibling models.
    let is_dir_entry = path.is_dir();
    let target = path.to_path_buf();
    let target_canon = target.canonicalize().unwrap_or_else(|_| target.clone());

    // Hard safety invariant: never delete the models root itself, nor
    // anything that isn't inside it.
    if target_canon == models_root_canon || !target_canon.starts_with(&models_root_canon) {
        return Err((axum::http::StatusCode::BAD_REQUEST, "Refusing to delete the models root or a path outside it".into()));
    }

    if is_dir_entry {
        std::fs::remove_dir_all(&target)
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    } else {
        std::fs::remove_file(&target)
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        if let Some(mmproj) = payload.mmproj_path.as_ref() {
            let mmproj_path = std::path::Path::new(mmproj);
            if mmproj_path.is_file() {
                let mmproj_canon = mmproj_path.canonicalize().unwrap_or_else(|_| mmproj_path.to_path_buf());
                if mmproj_canon.starts_with(&models_root_canon) {
                    let _ = std::fs::remove_file(mmproj_path);
                }
            }
        }
    }

    Ok(Json(ModelActionResponse { status: "deleted" }))
}

/// Composite key for `hf_downloads`/`hf_cancel` maps — filename alone can
/// collide across different repos, so key on both.
fn download_key(repo: &str, filename: &str) -> String {
    format!("{repo}::{filename}")
}

/// Turn a HuggingFace repo id (`org/model`) into a filesystem-safe folder name.
fn sanitize_repo_id(repo: &str) -> String {
    repo.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => c,
            _ => '_',
        })
        .collect()
}

async fn run_hf_download(
    repo: &str,
    filename: &str,
    target_path: &std::path::Path,
    downloads_map: &Arc<tokio::sync::Mutex<HashMap<String, DownloadProgress>>>,
    cancel_flag: &Arc<AtomicBool>,
    hf_token: Option<String>,
) -> Result<(), String> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    let key = download_key(repo, filename);

    if let Some(parent) = target_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| e.to_string())?;
    }

    let client = reqwest::Client::builder()
        .user_agent("dispos-ai/0.1")
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .http2_initial_stream_window_size(16 * 1024 * 1024)
        .http2_initial_connection_window_size(16 * 1024 * 1024)
        .build()
        .map_err(|e| e.to_string())?;

    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    let mut req = client.get(&url);
    if let Some(ref t) = hf_token {
        req = req.header("Authorization", format!("Bearer {}", t));
    }
    let initial = req
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let response = if initial.status().is_redirection() {
        let location = initial
            .headers()
            .get("location")
            .and_then(|v| v.to_str().ok())
            .ok_or("redirect without Location header")?
            .to_string();
        let base = reqwest::Url::parse(&url).map_err(|e| e.to_string())?;
        let redirect_url = base.join(&location).map_err(|e| e.to_string())?;
        reqwest::Client::builder()
            .user_agent("dispos-ai/0.1")
            .no_proxy()
            .http2_initial_stream_window_size(16 * 1024 * 1024)
            .http2_initial_connection_window_size(16 * 1024 * 1024)
            .build()
            .map_err(|e| e.to_string())?
            .get(redirect_url.as_str())
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?
    } else {
        initial.error_for_status().map_err(|e| e.to_string())?
    };

    let total_bytes = response.content_length();
    {
        let mut downloads = downloads_map.lock().await;
        if let Some(entry) = downloads.get_mut(&key) {
            entry.total_bytes = total_bytes;
        }
    }

    let mut file = tokio::io::BufWriter::with_capacity(
        1024 * 1024,
        tokio::fs::File::create(target_path)
            .await
            .map_err(|e| e.to_string())?,
    );

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_reported: std::time::Instant = std::time::Instant::now();
    while let Some(chunk) = stream.next().await {
        if cancel_flag.load(AtomicOrdering::Relaxed) {
            drop(file);
            let _ = tokio::fs::remove_file(target_path).await;
            let mut downloads = downloads_map.lock().await;
            if let Some(entry) = downloads.get_mut(&key) {
                entry.status = "cancelled".to_string();
            }
            return Ok(());
        }
        let chunk = chunk.map_err(|e| e.to_string())?;
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;
        if last_reported.elapsed() >= std::time::Duration::from_millis(250) {
            let mut downloads = downloads_map.lock().await;
            if let Some(entry) = downloads.get_mut(&key) {
                entry.downloaded_bytes = downloaded;
            }
            last_reported = std::time::Instant::now();
        }
    }
    file.flush().await.map_err(|e| e.to_string())?;

    let mut downloads = downloads_map.lock().await;
    if let Some(entry) = downloads.get_mut(&key) {
        entry.downloaded_bytes = downloaded;
        entry.status = "complete".to_string();
    }
    Ok(())
}

async fn hf_download_status(
    State(state): State<AppState>,
) -> Json<HashMap<String, DownloadProgress>> {
    let downloads = state.hf_downloads.lock().await;
    Json(downloads.clone())
}

#[derive(Debug, Deserialize)]
struct HfDownloadCancelRequest {
    repo: String,
    filename: String,
}

#[derive(Debug, Serialize)]
struct HfDownloadCancelResponse {
    status: &'static str,
}

async fn hf_download_cancel(
    State(state): State<AppState>,
    Json(payload): Json<HfDownloadCancelRequest>,
) -> Json<HfDownloadCancelResponse> {
    let key = download_key(&payload.repo, &payload.filename);
    let cancel_map = state.hf_cancel.lock().await;
    match cancel_map.get(&key) {
        Some(flag) => {
            flag.store(true, AtomicOrdering::Relaxed);
            Json(HfDownloadCancelResponse { status: "cancelling" })
        }
        None => Json(HfDownloadCancelResponse { status: "not_found" }),
    }
}

/// Find llama-server binary (same logic as in llama-backend)
fn find_llama_server_binary() -> Option<std::path::PathBuf> {
    let lmstudio_dir = std::env::var("USERPROFILE")
        .map(|p| std::path::PathBuf::from(p).join(".lmstudio\\extensions\\backends"))
        .ok()?;
    if let Ok(entries) = std::fs::read_dir(lmstudio_dir) {
        for entry in entries.flatten() {
            let candidate = entry.path().join("llama-server.exe");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Generate a simple unique model ID using timestamp + port
fn new_model_id(port: u16) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    format!("mdl-{}-{}", ts, port)
}

/// How long to wait for `hf_transformers_server.py` to print "READY" on
/// startup. Loading a safetensors model with `transformers` (weight
/// deserialization + CUDA transfer) can take a while on first run.
const HF_TRANSFORMERS_STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// Spawn `scripts/hf_transformers_server.py` against the model's parent
/// directory (what `from_pretrained` expects) and block on a worker thread
/// until it prints "READY" on stdout or the startup timeout elapses. Mirrors
/// the READY handshake in `tts-backend::TtsBackend::spawn_server`.
async fn spawn_hf_transformers_server(
    model_path: &std::path::Path,
    port: u16,
) -> Result<std::process::Child, String> {
    let script_path = std::path::PathBuf::from("scripts").join("hf_transformers_server.py");
    if !script_path.exists() {
        return Err(format!(
            "hf_transformers_server.py not found at: {}",
            script_path.display()
        ));
    }
    // `model_path` may already be a model directory (e.g. a collapsed HF
    // folder from `list_detected_models`), or a single weight file whose
    // parent directory is what `from_pretrained` expects.
    let model_dir = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path
            .parent()
            .ok_or_else(|| format!("Model path '{}' has no parent directory", model_path.display()))?
            .to_path_buf()
    };

    tokio::task::spawn_blocking(move || {
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};
        use std::time::Instant;

        let mut cmd = Command::new("python");
        cmd.arg(&script_path)
            .arg("--model-path")
            .arg(&model_dir)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn hf_transformers_server.py: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture stdout of hf_transformers_server.py".to_string())?;

        let start = Instant::now();
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    let _ = child.wait();
                    return Err("hf_transformers_server.py exited before signalling READY".to_string());
                }
                Ok(_) => {
                    if line.trim() == "READY" {
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("Error reading hf_transformers_server.py stdout: {}", e));
                }
            }
            if start.elapsed() > HF_TRANSFORMERS_STARTUP_TIMEOUT {
                let _ = child.kill();
                return Err("Timed out waiting for hf_transformers_server.py to start".to_string());
            }
        }

        Ok(child)
    })
    .await
    .map_err(|e| format!("spawn_blocking join error: {}", e))?
}

/// Load a model: detect modality and spawn the appropriate backend.
pub(crate) async fn set_studio_models(
    State(state): State<AppState>,
    Json(payload): Json<StudioModelsRequest>,
) -> Json<serde_json::Value> {
    let mut studio_models = state.studio_models.lock().await;
    studio_models.clear();
    studio_models.extend(payload.paths);
    Json(serde_json::json!({ "ok": true, "count": studio_models.len() }))
}

pub(crate) async fn set_media_retention(
    State(state): State<AppState>,
    Json(payload): Json<MediaRetentionConfig>,
) -> Json<MediaRetentionConfig> {
    state.media_store.set_retention(payload.ttl_seconds, payload.persist_disk);
    let (ttl_seconds, persist_disk) = state.media_store.retention();
    Json(MediaRetentionConfig { ttl_seconds, persist_disk })
}

pub(crate) async fn get_media_retention(
    State(state): State<AppState>,
) -> Json<MediaRetentionConfig> {
    let (ttl_seconds, persist_disk) = state.media_store.retention();
    Json(MediaRetentionConfig { ttl_seconds, persist_disk })
}

pub(crate) async fn load_model(
    State(state): State<AppState>,
    Json(payload): Json<ModelLoadRequest>,
) -> Result<Json<LoadedModelEntry>, (axum::http::StatusCode, String)> {
    let model_path = std::path::PathBuf::from(&payload.model_path);
    if !model_path.exists() {
        return Err((axum::http::StatusCode::BAD_REQUEST,
            format!("Model file not found at: {}", payload.model_path)));
    }

    let gpu_layers = payload.gpu_layers.unwrap_or(99);
    let context_size = payload.context_size.unwrap_or(4096);
    // `model_path` may be a collapsed HF model directory (see
    // `hf_model_dir_entry`) rather than a single weight file; classify those
    // the same way the catalog scan does before falling back to the
    // extension/content-based file classifier.
    let dir_name = model_path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_ascii_lowercase();
    let task_tags = if model_path.is_dir() {
        if let Some(hint) = read_task_tags_hint(&model_path) {
            hint
        } else if is_mesh3d_name(&dir_name) {
            crate::task_tags::mesh_tags_from_kinds(detect_mesh_input_kinds(&model_path))
        } else if model_path.join("model_index.json").is_file() {
            vec!["text-to-image".to_string()]
        } else if model_path.join("config.json").is_file() {
            crate::task_tags::local_heuristic_tags_for_dir(&model_path)
        } else {
            crate::task_tags::local_heuristic_tags_for_file(&model_path)
        }
    } else {
        model_path
            .parent()
            .and_then(read_task_tags_hint)
            .unwrap_or_else(|| crate::task_tags::local_heuristic_tags_for_file(&model_path))
    };
    tracing::info!("Detected task tags {:?} for model: {}", task_tags, model_path.display());
    let backend_category = crate::task_tags::backend_category_for_tags(&task_tags);

    // Allocate a new port
    let port = state.next_port.fetch_add(1, AtomicOrdering::SeqCst);
    let model_id = new_model_id(port);

    // Set when a text/vision model is launched with an mmproj projector, so the
    // loaded entry can advertise image input even for models detected as "text".
    let mut mmproj_loaded = false;

    match backend_category {
        Some(Modality::AudioTts) => {
            tracing::info!("Loading TTS model via TTS backend: {}", model_path.display());
            let backend_arc = state.registry.get_backend(Modality::AudioTts).await;
            if let Some(backend_arc) = backend_arc {
                let mut b = backend_arc.write().await;
                let opts = backend_trait::LoadOptions {
                    gpu_layers: Some(gpu_layers),
                    context_size: Some(context_size),
                    ..Default::default()
                };
                if let Err(e) = b.load_model(&model_path, &opts).await {
                    tracing::warn!("TTS backend load_model error: {}", e);
                    return Err((axum::http::StatusCode::BAD_REQUEST, e.to_string()));
                }
            } else {
                tracing::warn!("No TTS backend registered; model registered but inference will fail");
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Some(Modality::AudioAsr) => {
            tracing::info!("Loading ASR model via whisper backend: {}", model_path.display());
            let backend_arc = state.registry.get_backend(Modality::AudioAsr).await;
            if let Some(backend_arc) = backend_arc {
                let mut b = backend_arc.write().await;
                let opts = backend_trait::LoadOptions {
                    gpu_layers: Some(gpu_layers),
                    context_size: Some(context_size),
                    ..Default::default()
                };
                if let Err(e) = b.load_model(&model_path, &opts).await {
                    tracing::warn!("ASR backend load_model error: {}", e);
                    return Err((axum::http::StatusCode::BAD_REQUEST, e.to_string()));
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Some(Modality::Image) => {
            tracing::info!("Loading image model via SD backend: {}", model_path.display());
            let backend_arc = state.registry.get_backend(Modality::Image).await;
            if let Some(backend_arc) = backend_arc {
                let mut b = backend_arc.write().await;
                let opts = backend_trait::LoadOptions {
                    gpu_layers: Some(gpu_layers),
                    context_size: Some(context_size),
                    ..Default::default()
                };
                if let Err(e) = b.load_model(&model_path, &opts).await {
                    tracing::warn!("Image backend load_model error: {}", e);
                    return Err((axum::http::StatusCode::BAD_REQUEST, e.to_string()));
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Some(Modality::Video) => {
            tracing::info!("Loading video model via video backend: {}", model_path.display());
            let backend_arc = state.registry.get_backend(Modality::Video).await;
            if let Some(backend_arc) = backend_arc {
                let mut b = backend_arc.write().await;
                let opts = backend_trait::LoadOptions {
                    gpu_layers: Some(gpu_layers),
                    context_size: Some(context_size),
                    ..Default::default()
                };
                if let Err(e) = b.load_model(&model_path, &opts).await {
                    tracing::warn!("Video backend load_model error: {}", e);
                    return Err((axum::http::StatusCode::BAD_REQUEST, e.to_string()));
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Some(Modality::Mesh3D) => {
            tracing::info!("Loading 3D mesh model via mesh3d backend: {}", model_path.display());
            let backend_arc = state.registry.get_backend(Modality::Mesh3D).await;
            if let Some(backend_arc) = backend_arc {
                let mut b = backend_arc.write().await;
                let opts = backend_trait::LoadOptions {
                    gpu_layers: Some(gpu_layers),
                    context_size: Some(context_size),
                    ..Default::default()
                };
                if let Err(e) = b.load_model(&model_path, &opts).await {
                    tracing::warn!("Mesh3D backend load_model error: {}", e);
                    return Err((axum::http::StatusCode::BAD_REQUEST, e.to_string()));
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        _ => {
            // A directory here is a collapsed HF model dir with its own
            // `config.json` (the `model_index.json` / image case is routed
            // through the Image arm above via task-tag detection). Spawn
            // the transformers server directly against it.
            let is_safetensors = if model_path.is_dir() {
                true
            } else {
                model_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("safetensors"))
                    .unwrap_or(false)
            };

            if is_safetensors {
                // Config-detected CausalLM safetensors model — spawn the
                // transformers-backed OpenAI-compatible server instead of
                // llama-server. It speaks the same /v1/chat/completions
                // contract, so the existing chat proxy needs no changes.
                match spawn_hf_transformers_server(&model_path, port).await {
                    Ok(child) => {
                        state.children.register(model_id.clone(), child);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to start hf_transformers_server.py for '{}': {}",
                            model_path.display(),
                            e
                        );
                        return Err((axum::http::StatusCode::BAD_REQUEST, e));
                    }
                }
            } else if let Some(server_bin) = find_llama_server_binary() {
                let mut child_cmd = std::process::Command::new(&server_bin);
                child_cmd
                    .arg("-m").arg(&model_path)
                    .arg("--port").arg(port.to_string())
                    .arg("-ngl").arg(gpu_layers.to_string())
                    .arg("-c").arg(context_size.to_string())
                    .arg("--host").arg("127.0.0.1");
                tracing::info!("load_model: model_path={}, parent={:?}, payload.mmproj_path={:?}",
                    model_path.display(), model_path.parent(), payload.mmproj_path);
                let mmproj = payload.mmproj_path
                    .as_ref()
                    .filter(|s| !s.is_empty())
                    .map(std::path::PathBuf::from)
                    .filter(|p| {
                        let exists = p.exists();
                        tracing::info!("Provided mmproj_path exists={}: {}", exists, p.display());
                        exists
                    })
                    .or_else(|| {
                        let result = find_sibling_mmproj(&model_path);
                        tracing::info!("find_sibling_mmproj result: {:?}", result);
                        result
                    });
                if let Some(mmproj_path) = &mmproj {
                    tracing::info!("Using vision projector: {}", mmproj_path.display());
                    child_cmd.arg("--mmproj").arg(mmproj_path);
                    mmproj_loaded = true;
                } else {
                    tracing::warn!("No vision projector found for model: {}", model_path.display());
                }
                let child = child_cmd.spawn();
                match child {
                    Ok(c) => {
                        state.children.register(model_id.clone(), c);
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to spawn llama-server on port {}: {}", port, e);
                    }
                }
            } else {
                tracing::info!("llama-server binary not found; registering model in simulation mode on port {}", port);
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        }
    }

    let is_mesh3d = task_tags.iter().any(|t| t == "text-to-3d" || t == "image-to-3d");
    let mesh_input_kinds = if is_mesh3d { detect_mesh_input_kinds(&model_path) } else { None };

    let entry = LoadedModelEntry {
        model_id: model_id.clone(),
        model_path: payload.model_path.clone(),
        gpu_layers,
        context_size,
        port,
        image_input_available: crate::task_tags::image_input_available(&task_tags) || mmproj_loaded,
        mesh_input_kinds,
        task_tags,
    };

    {
        let mut models = state.loaded_models.lock().await;
        models.insert(model_id, entry.clone());
    }

    // Keep backwards-compat active_model pointing at this (most recently loaded) model
    {
        let mut status = state.active_model.lock().await;
        *status = LoadedModelStatus {
            is_loaded: true,
            model_path: Some(payload.model_path),
            gpu_layers: Some(gpu_layers),
            context_size: Some(context_size),
        };
    }

    Ok(Json(entry))
}

/// Unload a model by model_id. Kills its llama-server process.
async fn unload_model(
    State(state): State<AppState>,
    Json(payload): Json<ModelUnloadRequest>,
) -> Result<Json<Vec<LoadedModelEntry>>, (axum::http::StatusCode, String)> {
    if let Some(mid) = &payload.model_id {
        // Targeted unload: remove from registry and kill port via taskkill
        let entry = {
            let mut models = state.loaded_models.lock().await;
            models.remove(mid)
        };
        if let Some(e) = entry {
            // `take` removes the entry AND terminates the tracked child via
            // Child::kill() (see ChildRegistry::take). No shell-out, no PID
            // guessing — the Child handle is the source of truth.
            let _ = state.children.take(mid);
            // Belt-and-suspenders: kill whichever llama-server is listening on
            // that port in case the tracker is out of sync (e.g. process
            // adopted from a previous run).
            let _ = std::process::Command::new("powershell")
                .args(["-Command", &format!(
                    "$pid = (Get-NetTCPConnection -LocalPort {} -ErrorAction SilentlyContinue).OwningProcess; if ($pid) {{ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }}",
                    e.port
                )])
                .spawn();
            // Non-llama-server modalities (image/mesh3d/tts/audio/video) don't
            // register a child in `state.children` at load time — their
            // process is owned by the backend itself. Tear it down via the
            // backend trait so VRAM/metrics don't keep reporting it as loaded.
            if let Some(backend_modality) = crate::task_tags::backend_category_for_tags(&e.task_tags) {
                if let Some(backend_arc) = state.registry.get_backend(backend_modality).await {
                    let mut b = backend_arc.write().await;
                    if let Err(err) = b.unload_model().await {
                        tracing::warn!("unload_model: backend unload for '{}' ({:?}) failed: {}", mid, e.task_tags, err);
                    }
                } else {
                    tracing::warn!("unload_model: no backend registered for task tags '{:?}' (model '{}')", e.task_tags, mid);
                }
            }
        }
    } else {
        // Legacy unload: kill everything via the backend trait
        let backend_arc = state
            .registry
            .get_backend(Modality::Text)
            .await
            .ok_or((axum::http::StatusCode::SERVICE_UNAVAILABLE, "No text backend registered".to_string()))?;
        let mut b = backend_arc.write().await;
        b.unload_model().await.map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let mut models = state.loaded_models.lock().await;
        models.clear();
    }

    // Update backwards-compat status
    {
        let models = state.loaded_models.lock().await;
        let mut status = state.active_model.lock().await;
        if let Some(entry) = models.values().next() {
            *status = LoadedModelStatus {
                is_loaded: true,
                model_path: Some(entry.model_path.clone()),
                gpu_layers: Some(entry.gpu_layers),
                context_size: Some(entry.context_size),
            };
        } else {
            *status = LoadedModelStatus { is_loaded: false, model_path: None, gpu_layers: None, context_size: None };
        }
    }

    let models = state.loaded_models.lock().await;
    Ok(Json(models.values().cloned().collect()))
}
