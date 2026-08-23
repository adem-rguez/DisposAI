//! Static table describing which split-checkpoint *component kinds* each
//! diffusion-model architecture family needs when `sd.exe` is handed a
//! diffusion-transformer-only GGUF (no VAE / no text encoders bundled in).
//!
//! Matching is keyed on filename substrings for the *architecture family*
//! only (e.g. "flux2", "z-image") — never on a specific HF repo name. New
//! families are added here, not as special cases elsewhere.

/// A required split-checkpoint component. The `&'static str` on `Llm`/`Vae`
/// is a fuzzy filename hint (e.g. `"mistral"`, `"qwen3"`, `"flux2-vae"`) used
/// only to find a plausible candidate file — not an exact filename to match.
/// `Vae` is hinted (rather than a bare `"vae"` substring) because VAEs are
/// NOT interchangeable across incompatible architectures sharing the same
/// shared/vae folder — e.g. FLUX.2's VAE has 32 latent channels vs FLUX.1's
/// 16, so a generic "vae" match would silently pick the wrong file and fail
/// at sd.exe's tensor-shape validation instead of prompting a download.
/// Families that really do share one VAE (flux1/z-image/chroma) use the same
/// hint so they keep resolving to that one shared file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentKind {
    ClipL,
    ClipG,
    T5xxl,
    Llm(&'static str),
    Vae(&'static str),
}

impl ComponentKind {
    /// Human-readable name used in error messages.
    pub fn display_name(&self) -> &'static str {
        match self {
            ComponentKind::ClipL => "clip_l text encoder",
            ComponentKind::ClipG => "clip_g text encoder",
            ComponentKind::T5xxl => "t5xxl text encoder",
            ComponentKind::Llm(_) => "llm text encoder",
            ComponentKind::Vae(_) => "vae",
        }
    }

    /// Filename substrings (case-insensitive) that identify a candidate file
    /// as satisfying this component kind.
    pub fn filename_patterns(&self) -> Vec<&'static str> {
        match self {
            ComponentKind::ClipL => vec!["clip_l", "clip-l"],
            ComponentKind::ClipG => vec!["clip_g", "clip-g"],
            ComponentKind::T5xxl => vec!["t5xxl", "t5-xxl"],
            ComponentKind::Llm(hint) => vec![hint],
            ComponentKind::Vae(hint) => vec![hint],
        }
    }

    /// Shared-components subfolder (relative to the models root) this kind
    /// should be looked for in when it isn't next to the diffusion model.
    pub fn shared_subdir(&self) -> &'static str {
        match self {
            ComponentKind::Vae(_) => "shared/vae",
            _ => "shared/text_encoders",
        }
    }
}

/// A known default HuggingFace download source for a missing component.
/// `target_filename` is the destination filename after download (may differ
/// from `filename` when the upstream file needs renaming to satisfy
/// `ComponentKind::filename_patterns()`, e.g. BFL's `ae.safetensors` ->
/// something containing "vae").
#[derive(Debug, Clone, Copy)]
pub struct DefaultSource {
    pub repo: &'static str,
    pub filename: &'static str,
    pub target_filename: &'static str,
}

/// Everything `sd.exe` needs to know to run a given architecture family in
/// component (split-checkpoint) mode.
pub struct ComponentSpec {
    /// Family name, used only for logging/error messages.
    pub family: &'static str,
    pub components: &'static [ComponentKind],
    /// `--vae-format` value, if this family needs one specified explicitly.
    pub vae_format: Option<&'static str>,
    /// Extra `--model-args k=v,...` this family needs (e.g. Chroma/Qwen-Image
    /// specific flags). `None` means no `--model-args` flag is passed.
    pub model_args: Option<&'static str>,
    /// Applied to `ImageParams.cfg_scale` only when the caller didn't set one.
    pub default_cfg_scale: Option<f32>,
    /// Applied to `ImageParams.steps` only when the caller didn't set one.
    pub default_steps: Option<u32>,
    /// Known default download sources for specific components, where the
    /// real upstream HF repo is confidently known. Not populated for every
    /// `ComponentKind` — missing entries just mean "no known auto-download
    /// source yet".
    pub default_source: Option<&'static [(ComponentKind, DefaultSource)]>,
}

impl ComponentSpec {
    /// Look up the known default download source for `kind`, if any.
    pub fn default_source_for(&self, kind: &ComponentKind) -> Option<DefaultSource> {
        self.default_source?
            .iter()
            .find(|(k, _)| k == kind)
            .map(|(_, src)| *src)
    }
}

struct FamilyEntry {
    /// Filename substrings (case-insensitive). Order across `FAMILIES`
    /// matters: more specific patterns (e.g. flux2) must be listed before
    /// broader ones that would otherwise shadow them (e.g. flux1's "flux").
    patterns: &'static [&'static str],
    spec: ComponentSpec,
}

static FAMILIES: &[FamilyEntry] = &[
    // Flux.2-klein — a distilled variant with its own (much smaller) Qwen3
    // text encoder (4B for klein-4B, 8B for klein-9B), NOT the dev/pro/flex
    // Mistral-24B encoder. Must come before both the flux2 and flux1
    // fallbacks below, since klein filenames also contain "flux2"/"flux-2".
    // See leejet/stable-diffusion.cpp docs/flux2.md and diffusers'
    // Flux2KleinPipeline (text_encoder: Qwen3ForCausalLM).
    FamilyEntry {
        patterns: &["klein"],
        spec: ComponentSpec {
            family: "flux2-klein",
            components: &[ComponentKind::Llm("qwen3"), ComponentKind::Vae("flux2-vae")],
            vae_format: Some("flux2"),
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
            default_source: Some(&[
                (
                    ComponentKind::Llm("qwen3"),
                    DefaultSource {
                        repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                        filename: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                        target_filename: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                    },
                ),
                (
                    ComponentKind::Vae("flux2-vae"),
                    DefaultSource {
                        repo: "black-forest-labs/FLUX.2-dev",
                        filename: "ae.safetensors",
                        target_filename: "flux2-vae.safetensors",
                    },
                ),
            ]),
        },
    },
    // Flux.2 (dev/pro/flex) — must come after flux2-klein and before the
    // Flux1 fallback below.
    FamilyEntry {
        patterns: &["flux-2", "flux2", "flux.2"],
        spec: ComponentSpec {
            family: "flux2",
            // FLUX.2 [dev/pro/flex] replaces FLUX.1's clip_l/t5xxl pair with
            // a single Mistral text encoder (Mistral-Small-3.2-24B-Instruct-
            // 2506) — see black-forest-labs/flux2 on GitHub.
            components: &[ComponentKind::Llm("mistral"), ComponentKind::Vae("flux2-vae")],
            vae_format: Some("flux2"),
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
            default_source: Some(&[
                (
                    ComponentKind::Llm("mistral"),
                    DefaultSource {
                        repo: "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                        filename: "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
                        target_filename: "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
                    },
                ),
                (
                    ComponentKind::Vae("flux2-vae"),
                    DefaultSource {
                        repo: "black-forest-labs/FLUX.2-dev",
                        filename: "ae.safetensors",
                        target_filename: "flux2-vae.safetensors",
                    },
                ),
            ]),
        },
    },
    // Z-Image (reuses Flux1's VAE format).
    FamilyEntry {
        patterns: &["z-image", "z_image"],
        spec: ComponentSpec {
            family: "z-image",
            components: &[ComponentKind::Llm("qwen3"), ComponentKind::Vae("flux-vae")],
            vae_format: Some("flux"),
            model_args: None,
            default_cfg_scale: Some(1.0),
            default_steps: Some(8),
            default_source: Some(&[
                (
                    ComponentKind::Llm("qwen3"),
                    DefaultSource {
                        repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                        filename: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                        target_filename: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                    },
                ),
                (
                    ComponentKind::Vae("flux-vae"),
                    DefaultSource {
                        repo: "black-forest-labs/FLUX.1-schnell",
                        filename: "ae.safetensors",
                        target_filename: "flux-vae.safetensors",
                    },
                ),
            ]),
        },
    },
    // Qwen-Image.
    FamilyEntry {
        patterns: &["qwen-image"],
        spec: ComponentSpec {
            family: "qwen-image",
            components: &[ComponentKind::Llm("qwen"), ComponentKind::Vae("vae")],
            vae_format: None,
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
            default_source: None,
        },
    },
    // SD3 / SD3.5.
    FamilyEntry {
        patterns: &["sd3"],
        spec: ComponentSpec {
            family: "sd3",
            components: &[ComponentKind::ClipL, ComponentKind::ClipG, ComponentKind::T5xxl, ComponentKind::Vae("vae")],
            vae_format: Some("sd3"),
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
            default_source: None,
        },
    },
    // Chroma (Flux1-derived).
    FamilyEntry {
        patterns: &["chroma"],
        spec: ComponentSpec {
            family: "chroma",
            components: &[ComponentKind::T5xxl, ComponentKind::Vae("flux-vae")],
            vae_format: Some("flux"),
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
            default_source: Some(&[(
                ComponentKind::Vae("flux-vae"),
                DefaultSource {
                    repo: "black-forest-labs/FLUX.1-schnell",
                    filename: "ae.safetensors",
                    target_filename: "flux-vae.safetensors",
                },
            )]),
        },
    },
    // Flux1 fallback — must come after flux2's patterns above.
    FamilyEntry {
        patterns: &["flux"],
        spec: ComponentSpec {
            family: "flux1",
            components: &[ComponentKind::ClipL, ComponentKind::T5xxl, ComponentKind::Vae("flux-vae")],
            vae_format: Some("flux"),
            model_args: None,
            default_cfg_scale: None,
            default_steps: None,
            default_source: Some(&[(
                ComponentKind::Vae("flux-vae"),
                DefaultSource {
                    repo: "black-forest-labs/FLUX.1-schnell",
                    filename: "ae.safetensors",
                    target_filename: "flux-vae.safetensors",
                },
            )]),
        },
    },
];

/// Match a diffusion-model filename against the architecture family table.
/// Returns `None` if no family pattern is recognized.
pub fn match_component_spec(filename: &str) -> Option<&'static ComponentSpec> {
    let lower = filename.to_ascii_lowercase();
    FAMILIES
        .iter()
        .find(|entry| entry.patterns.iter().any(|p| lower.contains(p)))
        .map(|entry| &entry.spec)
}
