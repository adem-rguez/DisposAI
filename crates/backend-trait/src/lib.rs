use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use futures::Stream;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    AudioAsr,
    AudioTts,
    Embedding,
    Video,
    Mesh3D,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramEstimate {
    pub required_bytes: u64,
    pub recommended_gpu_layers: u32,
    pub total_layers: u32,
    pub fits_in_vram: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadOptions {
    pub gpu_layers: Option<u32>,
    pub context_size: Option<u32>,
    pub batch_size: Option<u32>,
    pub threads: Option<u32>,
    /// Optional mmproj (vision projector) path to enable image input on a
    /// text/vision GGUF. `None` falls back to sibling auto-detection.
    pub mmproj_path: Option<String>,
    pub params: std::collections::HashMap<String, String>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            gpu_layers: Some(99), // Default: offload all layers if possible
            context_size: Some(4096),
            batch_size: Some(512),
            threads: Some(8),
            mmproj_path: None,
            params: std::collections::HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_tokens: u32,
    pub stop_sequences: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 512,
            stop_sequences: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments_delta: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_handle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_data: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Parameters specific to image-generation requests (Modality::Image). Every
/// field is optional so backends fall back to their own sensible defaults;
/// text/audio/video backends simply ignore this field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageParams {
    pub negative_prompt: Option<String>,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub seed: Option<i64>,
    pub strength: Option<f32>,
}

/// Parameters specific to 3D-mesh-generation requests (Modality::Mesh3D).
/// Every field is optional so backends fall back to their own sensible
/// defaults; text/image/audio/video backends simply ignore this field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Mesh3dParams {
    pub input_kind: Option<String>,
    pub images: Option<Vec<Vec<u8>>>,
    pub steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub seed: Option<i64>,
    pub output_format: Option<String>,
    pub texture: Option<bool>,
    pub foreground_ratio: Option<f32>,
    /// Any adapter-specific parameters not covered by the named fields above
    /// (e.g. SF3D's `remesh_option`/`target_vertex_count`), passed straight
    /// through to `threed_server.py` untouched. Populated from
    /// `Mesh3dGenerationRequest`'s `#[serde(flatten)]` `extra` map.
    #[serde(default)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

/// Parameters specific to speech-synthesis requests (Modality::AudioTts).
/// Optional so backends fall back to their own defaults; other backends
/// ignore this field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioParams {
    pub speed: Option<f32>,
}

/// Parameters specific to video-generation requests (Modality::Video). Every
/// field is optional so backends fall back to their own sensible defaults;
/// text/image/audio/mesh backends simply ignore this field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoParams {
    pub negative_prompt: Option<String>,
    /// Conditioning image for img2video pipelines (e.g. Stable Video
    /// Diffusion). Text-to-video pipelines simply ignore this.
    pub image: Option<Vec<u8>>,
    pub num_frames: Option<u32>,
    pub height: Option<u32>,
    pub width: Option<u32>,
    pub num_inference_steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub fps: Option<u32>,
    pub seed: Option<i64>,
    pub output_format: Option<String>,
    /// Any adapter-specific parameters not covered by the named fields above,
    /// passed straight through to `video_diffusers_server.py` untouched.
    #[serde(default)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt: String,
    /// Optional structured chat messages. When provided, backends that support
    /// /v1/chat/completions (e.g. llama-server) will use these directly with
    /// the model's native chat template (enables Qwen3 <think> reasoning).
    pub messages: Option<Vec<ChatMessage>>,
    pub sampling: SamplingParams,
    pub modality: Modality,
    pub image_input: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    /// Image-generation params (Modality::Image only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_params: Option<ImageParams>,
    /// 3D-mesh-generation params (Modality::Mesh3D only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mesh_params: Option<Mesh3dParams>,
    /// Speech-synthesis params (Modality::AudioTts only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_params: Option<AudioParams>,
    /// Video-generation params (Modality::Video only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_params: Option<VideoParams>,
    /// Cooperative cancellation flag. Backends that support cancelling an
    /// in-flight generation (e.g. sd-backend's `sd.exe` subprocess,
    /// llama-backend's HTTP request) poll this and return
    /// `BackendError::Cancelled` when it's set. Not (de)serialized — it's
    /// only ever set in-process by daemon-core when it registers a job.
    #[serde(skip)]
    pub cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub output_text: String,
    pub output_data: Option<Vec<u8>>,
    pub tokens_generated: u32,
    pub generation_time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Progress record for a long-running generation job (image/mesh/tts), shared
/// between Python inference servers, backends, and daemon-core's job map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationProgress {
    pub job_id: String,
    pub modality: String,
    pub phase: String,
    pub step: u32,
    pub total: u32,
    pub percent: f32,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Handle to the completed media (e.g. `/v1/media/<id>`), set once a job
    /// finishes successfully and produced output. Mirrors `ToolResult::media_handle`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_handle: Option<String>,
    /// MIME type of the completed media, e.g. `image/png`. Mirrors `ToolResult::media_type`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceChunk {
    pub request_id: String,
    pub delta_text: String,
    pub delta_data: Option<Vec<u8>>,
    pub is_final: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_tool_call: Option<ToolCallDelta>,
}

pub type InferenceStream = Pin<Box<dyn Stream<Item = Result<InferenceChunk, BackendError>> + Send>>;

/// A known default HuggingFace download source for a missing component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingComponentSource {
    pub repo: String,
    pub filename: String,
    pub target_filename: String,
}

/// One missing split-checkpoint component (e.g. VAE, text encoder), reported
/// so the frontend can offer a one-click download fix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingComponentInfo {
    pub kind_name: String,
    pub target_path: String,
    pub source: Option<MissingComponentSource>,
}

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Failed to load model: {0}")]
    LoadError(String),
    #[error("Inference execution error: {0}")]
    InferenceError(String),
    #[error("Insufficient VRAM: required {required_bytes} B, available {available_bytes} B")]
    OutofVram { required_bytes: u64, available_bytes: u64 },
    #[error("Unsupported modality: {0:?}")]
    UnsupportedModality(Modality),
    #[error("Missing required components")]
    MissingComponents(Vec<MissingComponentInfo>),
    #[error("Python environment not installed for backend: {0}")]
    EnvNotInstalled(String),
    #[error("Generation cancelled")]
    Cancelled,
    #[error("Backend dynamic error: {0}")]
    Other(String),
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Friendly name of this backend (e.g. "llama.cpp", "stable-diffusion.cpp")
    fn name(&self) -> &'static str;

    /// Supported model modalities
    fn supported_modalities(&self) -> &[Modality];

    /// Estimate VRAM usage for a given model file and load options
    async fn estimate_vram(
        &self,
        model_path: &Path,
        options: &LoadOptions,
    ) -> Result<VramEstimate, BackendError>;

    /// Load model into memory/GPU
    async fn load_model(
        &mut self,
        model_path: &Path,
        options: &LoadOptions,
    ) -> Result<(), BackendError>;

    /// Unload model from memory/GPU
    async fn unload_model(&mut self) -> Result<(), BackendError>;

    /// Check if a model is currently loaded in this backend
    fn is_loaded(&self) -> bool;

    /// Run full synchronous inference
    async fn generate(
        &mut self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, BackendError>;

    /// Run streaming inference
    async fn generate_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceStream, BackendError>;

    /// Poll for progress on a currently-running generation. Backends without
    /// a progress source (most of them) keep the default `None`.
    async fn poll_progress(&self) -> Option<GenerationProgress> {
        None
    }

    /// Fetch adapter parameter schemas (Mesh3D backends only), used to serve
    /// `GET /v1/models3d/schema` so a future UI can build a dynamic parameter
    /// form. Backends without an underlying schema source keep the default
    /// `None`. `&mut self` because implementations lazily spawn their Python
    /// subprocess here (schema is otherwise unavailable until the first
    /// `generate()` call).
    async fn get_param_schema(&mut self) -> Option<serde_json::Value> {
        None
    }
}
