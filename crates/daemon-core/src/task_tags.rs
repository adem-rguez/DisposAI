//! HuggingFace-style task-tag classification.
//!
//! Replaces the old single coarse `modality: String` field
//! (`"text"|"vision"|"image"|"video"|"audio"|"tts"|"mesh3d"`) with the model's
//! real task tag(s) (e.g. `"text-to-image"`, `"image-to-3d"`), supporting more
//! than one tag per model. `backend_trait::Modality` is unchanged and remains
//! the resolution target for "which physical backend process runs this".

use backend_trait::Modality;
use std::path::Path;

/// Every HF `pipeline_tag` this daemon recognizes, and which physical backend
/// category (if any) it dispatches to. `None` means the chat/llama-server
/// path — used for both plain text generation and vision-chat models.
const KNOWN_TASK_TAGS: &[(&str, Option<Modality>)] = &[
    // Chat / llama-server path (no dedicated backend)
    ("text-generation", None),
    ("text2text-generation", None),
    ("conversational", None),
    ("image-text-to-text", None),
    ("visual-question-answering", None),
    // Video
    ("text-to-video", Some(Modality::Video)),
    ("image-to-video", Some(Modality::Video)),
    ("video-to-video", Some(Modality::Video)),
    ("video-classification", Some(Modality::Video)),
    ("video-text-to-text", Some(Modality::Video)),
    // Image
    ("text-to-image", Some(Modality::Image)),
    ("image-to-image", Some(Modality::Image)),
    ("unconditional-image-generation", Some(Modality::Image)),
    ("image-classification", Some(Modality::Image)),
    ("image-segmentation", Some(Modality::Image)),
    ("zero-shot-image-classification", Some(Modality::Image)),
    // Mesh3D
    ("image-to-3d", Some(Modality::Mesh3D)),
    ("text-to-3d", Some(Modality::Mesh3D)),
    // Audio
    ("text-to-speech", Some(Modality::AudioTts)),
    ("automatic-speech-recognition", Some(Modality::AudioAsr)),
    ("audio-classification", Some(Modality::AudioAsr)),
];

/// Validate/pass through a raw HF `pipeline_tag`. Returns the canonical
/// allowlisted string if recognized, else `None`.
pub fn known_task_tag(raw: &str) -> Option<&'static str> {
    KNOWN_TASK_TAGS.iter().find(|(tag, _)| *tag == raw).map(|(tag, _)| *tag)
}

/// Resolve a model's task tag set to the physical backend category that
/// should run it. Priority: AudioTts > AudioAsr > Image > Video > Mesh3D >
/// `None` (chat/llama-server path). Unrecognized tags are ignored.
pub fn backend_category_for_tags(tags: &[String]) -> Option<Modality> {
    let categories: Vec<Modality> = tags
        .iter()
        .filter_map(|t| KNOWN_TASK_TAGS.iter().find(|(tag, _)| *tag == t.as_str()))
        .filter_map(|(_, modality)| *modality)
        .collect();

    for candidate in [Modality::AudioTts, Modality::AudioAsr, Modality::Image, Modality::Video, Modality::Mesh3D] {
        if categories.contains(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// True when the tag set means this model accepts image input on the
/// chat/llama-server path (native vision-chat model).
pub fn image_input_available(tags: &[String]) -> bool {
    tags.iter().any(|t| t == "image-text-to-text" || t == "visual-question-answering")
}

/// Turn a mesh3d model's accepted input kinds (subset of "text", "image",
/// "multi_image", as produced by `crate::http::detect_mesh_input_kinds`) into
/// task tags. Models accepting text input get "text-to-3d"; models accepting
/// image or multi-image input get "image-to-3d" (both can be present, e.g.
/// trellis/hunyuan3d).
pub fn mesh_tags_from_kinds(kinds: Option<Vec<String>>) -> Vec<String> {
    let kinds = kinds.unwrap_or_default();
    let mut tags = Vec::new();
    if kinds.iter().any(|k| k == "text") {
        tags.push("text-to-3d".to_string());
    }
    if kinds.iter().any(|k| k == "image" || k == "multi_image") {
        tags.push("image-to-3d".to_string());
    }
    if tags.is_empty() {
        tags.push("image-to-3d".to_string());
    }
    tags
}

/// Local (non-HF) keyword-based classification for a single model file,
/// mirroring the branches formerly in `detect_model_modality`.
pub fn local_heuristic_tags_for_file(path: &Path) -> Vec<String> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    // Match keywords against the parent directory name too: HF downloads land
    // in a repo-named folder while the file itself is often generically named.
    let file_name = {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let parent = path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or_default();
        format!("{parent} {name}").to_ascii_lowercase()
    };

    let image_tag = |file_name: &str| -> Vec<String> {
        if file_name.contains("edit") {
            vec!["image-to-image".to_string()]
        } else {
            vec!["text-to-image".to_string()]
        }
    };

    match extension.as_str() {
        "gguf" => {
            use std::io::Read;
            let mut bytes = Vec::new();
            let _ = std::fs::File::open(path)
                .and_then(|mut file| file.by_ref().take(2 * 1024 * 1024).read_to_end(&mut bytes));
            if bytes.windows(b"image-text-to-text".len()).any(|window| window == b"image-text-to-text") {
                vec!["image-text-to-text".to_string()]
            } else if file_name.contains("kokoro") || file_name.contains("tts") || file_name.contains("vits") || file_name.contains("bark") || file_name.contains("piper") {
                vec!["text-to-speech".to_string()]
            } else if file_name.contains("whisper") || file_name.contains("wav2vec") || file_name.contains("hubert") || file_name.contains("asr") {
                vec!["automatic-speech-recognition".to_string()]
            } else if file_name.contains("video") || file_name.contains("wan") || file_name.contains("mochi") {
                vec!["text-to-video".to_string()]
            } else if file_name.contains("stable") || file_name.contains("diffusion") || file_name.contains("sdxl") || file_name.contains("flux") || file_name.contains("vae") || file_name.contains("unet") || file_name.contains("qwen-image") || file_name.contains("image-edit") || file_name.contains("z-image") || file_name.contains("z_image") {
                image_tag(&file_name)
            } else if crate::http::is_mesh3d_name(&file_name) {
                mesh_tags_from_kinds(crate::http::detect_mesh_input_kinds(path))
            } else {
                vec!["text-generation".to_string()]
            }
        }
        "safetensors" if crate::http::detect_safetensors_causal_lm(path) => vec!["text-generation".to_string()],
        "safetensors" | "ckpt" => {
            if crate::http::is_mesh3d_name(&file_name) {
                mesh_tags_from_kinds(crate::http::detect_mesh_input_kinds(path))
            } else if file_name.contains("whisper") || file_name.contains("wav2vec") || file_name.contains("hubert") {
                vec!["automatic-speech-recognition".to_string()]
            } else if file_name.contains("kokoro") || file_name.contains("tts") || file_name.contains("vits") || file_name.contains("bark") {
                vec!["text-to-speech".to_string()]
            } else if file_name.contains("video") || file_name.contains("wan") || file_name.contains("mochi") {
                vec!["text-to-video".to_string()]
            } else {
                image_tag(&file_name)
            }
        }
        "pth" | "pt" => {
            if crate::http::is_mesh3d_name(&file_name) {
                mesh_tags_from_kinds(crate::http::detect_mesh_input_kinds(path))
            } else if file_name.contains("kokoro") || file_name.contains("tts") || file_name.contains("vits") || file_name.contains("bark") || file_name.contains("tacotron") || file_name.contains("fastspeech") {
                vec!["text-to-speech".to_string()]
            } else if file_name.contains("whisper") || file_name.contains("wav2vec") || file_name.contains("hubert") || file_name.contains("asr") {
                vec!["automatic-speech-recognition".to_string()]
            } else if file_name.contains("stable") || file_name.contains("diffusion") || file_name.contains("vae") || file_name.contains("unet") {
                image_tag(&file_name)
            } else if file_name.contains("video") || file_name.contains("wan") || file_name.contains("mochi") {
                vec!["text-to-video".to_string()]
            } else {
                vec!["text-generation".to_string()]
            }
        }
        "bin" => {
            if file_name.contains("kokoro") || file_name.contains("tts") || file_name.contains("vits") || file_name.contains("piper") {
                vec!["text-to-speech".to_string()]
            } else {
                vec!["automatic-speech-recognition".to_string()]
            }
        }
        "onnx" => {
            if crate::http::is_mesh3d_name(&file_name) {
                mesh_tags_from_kinds(crate::http::detect_mesh_input_kinds(path))
            } else if file_name.contains("whisper") || file_name.contains("wav2vec") || file_name.contains("asr") {
                vec!["automatic-speech-recognition".to_string()]
            } else if file_name.contains("stable") || file_name.contains("diffusion") || file_name.contains("unet") || file_name.contains("vae") {
                image_tag(&file_name)
            } else if file_name.contains("video") || file_name.contains("wan") {
                vec!["text-to-video".to_string()]
            } else {
                vec!["text-to-speech".to_string()]
            }
        }
        "ot" | "tflite" | "mlmodel" | "engine" => vec!["text-generation".to_string()],
        "mp4" | "webm" => vec!["text-to-video".to_string()],
        _ => vec!["text-generation".to_string()],
    }
}

/// Local (non-HF) keyword-based classification for a HuggingFace-style model
/// directory (has its own `config.json`), mirroring the branches formerly in
/// `detect_hf_config_dir_modality`. Does not consult any cached hint file —
/// callers check that first and only fall back to this.
pub fn local_heuristic_tags_for_dir(dir: &Path) -> Vec<String> {
    let config_path = dir.join("config.json");
    let is_causal_lm = std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|contents| serde_json::from_str::<serde_json::Value>(&contents).ok())
        .map(|config| {
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
        })
        .unwrap_or(false);
    if is_causal_lm {
        return vec!["text-generation".to_string()];
    }
    let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_ascii_lowercase();
    if crate::http::is_mesh3d_name(&dir_name) {
        mesh_tags_from_kinds(crate::http::detect_mesh_input_kinds(dir))
    } else if dir_name.contains("whisper") || dir_name.contains("wav2vec") || dir_name.contains("hubert") {
        vec!["automatic-speech-recognition".to_string()]
    } else if dir_name.contains("kokoro") || dir_name.contains("tts") || dir_name.contains("vits") || dir_name.contains("bark") {
        vec!["text-to-speech".to_string()]
    } else if dir_name.contains("edit") {
        vec!["image-to-image".to_string()]
    } else {
        vec!["text-to-image".to_string()]
    }
}
