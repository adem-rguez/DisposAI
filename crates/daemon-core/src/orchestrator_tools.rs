//! Model-agnostic tools for the orchestrating text model.
//!
//! These let the orchestrator discover what is on the machine and drive any of it:
//!
//!   `list_models` — what exists, what modality it is, whether it is loaded
//!   `run_model`   — prompt one of them (loading it first if needed) and get the output back
//!
//! Both return their result to the orchestrator, which then decides what to do next.

use axum::extract::State;
use axum::Json;
use backend_trait::{
    GenerationProgress, InferenceRequest, Modality, SamplingParams, ToolCall, ToolResult, ToolSchema,
};
use tracing::info;

use crate::http::{list_detected_models, load_model, unload_model, AppState, ModelLoadRequest, ModelUnloadRequest};
use crate::profiler::{FitEstimationRequest, HardwareProfiler};

/// Schemas advertised to the orchestrator.
pub fn schemas() -> Vec<ToolSchema> {
    vec![
        ToolSchema {
            name: "list_models".into(),
            description: "List every model available on this machine, with its task tag(s) \
                          (e.g. text-generation, image-text-to-text, text-to-image, \
                          text-to-speech, automatic-speech-recognition, text-to-video, \
                          text-to-3d/image-to-3d) and whether it is currently loaded. Call this \
                          first when you need to know what you can run. If more than one model \
                          matches what the user wants and you are not confident which one they'd \
                          pick, call present_model_choice with the candidates instead of silently \
                          picking one yourself."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "task_tag": {
                        "type": "string",
                        "description": "Optional filter, e.g. 'text-to-speech' to list only speech models"
                    }
                }
            }),
        },
        ToolSchema {
            name: "run_model".into(),
            description: "Send a prompt to one of the models from list_models. The model is \
                          loaded automatically if it is not already. This call waits for \
                          generation to finish (up to several minutes for image/video/3D/speech \
                          models) and returns the result directly - you do not need to poll for \
                          it. A job_id is included for reference; use get_generation_progress \
                          only if you specifically want to check on or list jobs out of band, \
                          and cancel_job only if the user asks you to stop one. Media generated \
                          by a previous job can be chained into a later run_model call (e.g. \
                          image -> 3D mesh, or image -> image) by passing its handle as the \
                          'image' argument."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Name of the model to run, exactly as given by list_models"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to that model"
                    },
                    "image": {
                        "type": "string",
                        "description": "Optional media handle from a previous run_model result \
                                        (the '/v1/media/<id>' path, or just the id) to use as \
                                        image input, e.g. for image-to-3D or image-to-image."
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional generation parameters for this model, as \
                                        returned by inspect_model. e.g. {\"steps\":30,\"seed\":42} \
                                        for image/3D/video models, or {\"temperature\":0.8} for \
                                        text models."
                    }
                },
                "required": ["model", "prompt"]
            }),
        },
        ToolSchema {
            name: "present_model_choice".into(),
            description: "Hand model selection to the user instead of picking one yourself. \
                          Call this when more than one model from list_models could do what the \
                          user asked and you aren't confident which one they want. Do not call \
                          run_model after this - the user will pick, and the daemon will act on \
                          their choice."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Names of the candidate models, exactly as given by list_models."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional short note on what the user is trying to do, for context."
                    }
                },
                "required": ["models"]
            }),
        },
        ToolSchema {
            name: "inspect_model".into(),
            description: "Look up what generation parameters a model accepts before calling \
                          run_model with a 'params' object. Call this whenever you want to set \
                          something like steps, cfg_scale, seed or temperature. Requires the \
                          exact model name from list_models — call list_models first if you \
                          don't already have it verbatim."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Name of the model, exactly as given by list_models. \
                                        Do not guess this — call list_models first."
                    }
                },
                "required": ["model"]
            }),
        },
        ToolSchema {
            name: "get_generation_progress".into(),
            description: "Check the live status of an image, 3D, video or audio generation job \
                          by job_id, or omit job_id to list every active job. run_model already \
                          waits for its own job to finish and returns the result directly, so \
                          you normally don't need this - it's for checking on a job separately \
                          from the run_model call that started it (e.g. after starting several \
                          at once)."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job_id returned by an image/3D/video/audio generation request."
                    },
                    "wait_seconds": {
                        "type": "integer",
                        "description": "Optional. If provided and the job is still running, waits \
                                        up to this many seconds (server-side) before returning the \
                                        current status - lets you set your own check-back interval \
                                        instead of polling immediately. Clamped to 1-60. Ignored if \
                                        job_id is omitted."
                    }
                }
            }),
        },
        ToolSchema {
            name: "cancel_job".into(),
            description: "Cancel a running generation job by its job_id. Use this if a job has \
                          been checked via get_generation_progress and is taking too long or \
                          appears stuck. Cancellation is cooperative - the backend checks for it \
                          periodically, so it may take a moment to actually stop."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job_id to cancel, as returned by run_model."
                    }
                },
                "required": ["job_id"]
            }),
        },
    ]
}

/// Handle `tool_call` if it is one of ours. Returns `None` for anything else so the
/// caller can fall through to the per-backend dispatcher.
pub async fn dispatch(state: &AppState, tool_call: &ToolCall, job_id: &str, conversation_id: &str) -> Option<ToolResult> {
    match tool_call.name.as_str() {
        "list_models" => Some(list_models(state, tool_call).await),
        "present_model_choice" => Some(present_model_choice(state, tool_call).await),
        "run_model" => Some(run_model(state, tool_call, job_id, conversation_id).await),
        "inspect_model" => Some(inspect_model(state, tool_call).await),
        "get_generation_progress" => Some(get_generation_progress(state, tool_call).await),
        "cancel_job" => Some(cancel_job(state, tool_call).await),
        _ => None,
    }
}

fn arg<'a>(tool_call: &'a ToolCall, key: &str) -> Option<&'a str> {
    tool_call.arguments.get(key).and_then(|v| v.as_str())
}

fn arg_object<'a>(tool_call: &'a ToolCall, key: &str) -> Option<&'a serde_json::Value> {
    tool_call.arguments.get(key).filter(|v| v.is_object())
}

fn ok(tool_call: &ToolCall, content: String) -> ToolResult {
    ToolResult {
        tool_call_id: tool_call.id.clone(),
        content,
        media_handle: None,
        media_data: None,
        media_type: None,
        is_error: false,
    }
}

fn err(tool_call: &ToolCall, content: String) -> ToolResult {
    ToolResult {
        tool_call_id: tool_call.id.clone(),
        content,
        media_handle: None,
        media_data: None,
        media_type: None,
        is_error: true,
    }
}

/// A model as the orchestrator sees it: file name, modality, loaded or not.
async fn inventory(state: &AppState) -> Vec<serde_json::Value> {
    let loaded = state.loaded_models.lock().await.clone();
    let catalog = list_detected_models().await.0;
    let studio_paths = state.studio_models.lock().await.clone();

    let mut rows: Vec<serde_json::Value> = catalog
        .iter()
        .filter(|entry| studio_paths.iter().any(|p| paths_match(p, &entry.path)))
        .map(|entry| {
            let running = loaded.values().find(|l| paths_match(&l.model_path, &entry.path));
            serde_json::json!({
                "model": entry.name,
                "task_tags": running.map(|l| l.task_tags.clone()).unwrap_or_else(|| entry.task_tags.clone()),
                "loaded": running.is_some(),
            })
        })
        .collect();

    // Anything loaded from outside the models directory still needs to be listed.
    for entry in loaded.values() {
        let name = file_name_of(&entry.model_path);
        if !rows.iter().any(|r| r["model"] == serde_json::Value::String(name.clone())) {
            rows.push(serde_json::json!({
                "model": name,
                "task_tags": entry.task_tags,
                "loaded": true,
            }));
        }
    }
    rows
}

fn file_name_of(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn paths_match(a: &str, b: &str) -> bool {
    a.replace('\\', "/").eq_ignore_ascii_case(&b.replace('\\', "/"))
        || file_name_of(a).eq_ignore_ascii_case(&file_name_of(b))
}

async fn list_models(state: &AppState, tool_call: &ToolCall) -> ToolResult {
    let filter = arg(tool_call, "task_tag").map(|m| m.to_ascii_lowercase());
    let mut rows = inventory(state).await;
    if let Some(filter) = &filter {
        rows.retain(|r| {
            r["task_tags"]
                .as_array()
                .map(|tags| tags.iter().any(|t| t.as_str().unwrap_or("").eq_ignore_ascii_case(filter)))
                .unwrap_or(false)
        });
    }

    if rows.is_empty() {
        return ok(tool_call, "No models are available on this machine.".into());
    }
    ok(tool_call, serde_json::to_string(&rows).unwrap_or_else(|_| "[]".into()))
}

/// The orchestrator explicitly wants the user to pick between candidate models. This just
/// resolves the given names to their full rows (same shape as list_models); http.rs's
/// streaming loop is what actually turns this into a `model_choice` SSE event and stops
/// the orchestrator loop (unless autopilot is on, in which case it is fed back like any
/// other tool result and the orchestrator picks for itself).
async fn present_model_choice(state: &AppState, tool_call: &ToolCall) -> ToolResult {
    let Some(names) = tool_call.arguments.get("models").and_then(|v| v.as_array()) else {
        return err(tool_call, "present_model_choice needs a 'models' array.".into());
    };
    let names: Vec<String> = names.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
    if names.is_empty() {
        return err(tool_call, "present_model_choice needs at least one model name.".into());
    }

    let rows = inventory(state).await;
    let selected: Vec<serde_json::Value> = rows
        .into_iter()
        .filter(|r| names.iter().any(|n| r["model"].as_str() == Some(n.as_str())))
        .collect();
    if selected.is_empty() {
        return err(tool_call, "None of those models were found. Call list_models to get exact names.".into());
    }
    ok(tool_call, serde_json::to_string(&selected).unwrap_or_else(|_| "[]".into()))
}

fn summarize_progress(p: &GenerationProgress) -> String {
    format!(
        "{} job {}: {} ({}), step {}/{} ({:.0}%){}",
        p.modality,
        p.job_id,
        p.status,
        p.phase,
        p.step,
        p.total,
        p.percent,
        p.message.as_deref().map(|m| format!(" - {}", m)).unwrap_or_default()
    )
}

async fn get_generation_progress(state: &AppState, tool_call: &ToolCall) -> ToolResult {
    let Some(job_id) = arg(tool_call, "job_id").filter(|id| !id.is_empty()).map(|s| s.to_string()) else {
        let jobs = state.job_progress.lock().await;
        if jobs.is_empty() {
            return ok(tool_call, "No active jobs.".into());
        }
        let summary = jobs.values().map(summarize_progress).collect::<Vec<_>>().join("\n");
        return ok(tool_call, summary);
    };

    // Default to a 5s wait even if the model omits `wait_seconds` — otherwise a
    // forgetful model just re-checks instantly, burning tool hops every few
    // hundred ms instead of actually giving the job time to progress.
    const DEFAULT_WAIT_SECONDS: u64 = 5;
    let wait_seconds = tool_call
        .arguments
        .get("wait_seconds")
        .and_then(|v| v.as_i64())
        .map(|s| s.clamp(1, 60) as u64)
        .unwrap_or(DEFAULT_WAIT_SECONDS);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(wait_seconds);
    loop {
        let still_running = {
            let jobs = state.job_progress.lock().await;
            jobs.get(&job_id).map(|p| p.status == "running").unwrap_or(false)
        };
        if !still_running || std::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    let jobs = state.job_progress.lock().await;
    match jobs.get(&job_id) {
        Some(progress) => {
            let summary = summarize_progress(progress);
            let mut result = if progress.status == "error" {
                err(tool_call, summary)
            } else {
                ok(tool_call, summary)
            };
            result.media_handle = progress.media_handle.clone();
            result.media_type = progress.media_type.clone();
            result
        }
        None => err(tool_call, format!("No such job '{}'.", job_id)),
    }
}

async fn cancel_job(state: &AppState, tool_call: &ToolCall) -> ToolResult {
    let Some(job_id) = arg(tool_call, "job_id").filter(|id| !id.is_empty()) else {
        return err(tool_call, "cancel_job needs a 'job_id' argument.".into());
    };
    let cancel_map = state.job_cancel.lock().await;
    match cancel_map.get(job_id) {
        Some(flag) => {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
            ok(tool_call, format!(
                "Cancellation requested for job '{}'. It may take a moment to actually stop; \
                check get_generation_progress to confirm.", job_id
            ))
        }
        None => err(tool_call, format!("No such active job '{}'.", job_id)),
    }
}

async fn run_model(state: &AppState, tool_call: &ToolCall, job_id: &str, conversation_id: &str) -> ToolResult {
    let Some(target) = arg(tool_call, "model") else {
        return err(tool_call, "run_model needs a 'model' argument.".into());
    };
    let prompt = arg(tool_call, "prompt").unwrap_or("");
    let params = arg_object(tool_call, "params");

    // Resolve against what is already loaded, else load it from the catalog.
    let mut entry = find_loaded(state, target).await;
    let mut orchestrator_unloaded: Option<crate::http::LoadedModelEntry> = None;
    if entry.is_none() {
        let Some(found) = find_in_catalog(target).await else {
            return err(tool_call, format!(
                "No model named '{}'. Call list_models to see what is available.", target
            ));
        };

        // Decide, from a live VRAM probe, whether the orchestrator (the resident
        // text/vision model) needs to step aside before this load is attempted.
        let sys = HardwareProfiler::probe();
        let fit = HardwareProfiler::estimate_fit(
            &FitEstimationRequest {
                parameter_count_billions: 0.0,
                quantization: String::new(),
                context_size: 4096,
                modality: found.task_tags.first().cloned().unwrap_or_default(),
                model_size_bytes: Some(found.size_bytes),
            },
            &sys,
        );
        let required = fit.total_required_vram_bytes as f64;
        let free = sys.free_vram_bytes as f64;
        let looks_impossible = required > free * 1.15;

        info!("run_model: loading '{}' on demand", found.name);

        if looks_impossible {
            if let Some(orch) = find_orchestrator_entry(state).await {
                if !paths_match(&orch.model_path, &found.path) {
                    unload_model(
                        State(state.clone()),
                        Json(ModelUnloadRequest { model_id: Some(orch.model_id.clone()) }),
                    )
                    .await
                    .ok();
                    orchestrator_unloaded = Some(orch);
                }
            }
        }

        let mut loaded = load_model(
            State(state.clone()),
            Json(ModelLoadRequest {
                model_path: found.path.clone(),
                gpu_layers: None,
                context_size: None,
                mmproj_path: found.mmproj_path.clone(),
            }),
        )
        .await;

        // Close call (not clearly doable, not clearly impossible): try normally
        // first, and only unload the orchestrator to retry if that load fails.
        if loaded.is_err() && orchestrator_unloaded.is_none() && !looks_impossible {
            if let Some(orch) = find_orchestrator_entry(state).await {
                if !paths_match(&orch.model_path, &found.path) {
                    unload_model(
                        State(state.clone()),
                        Json(ModelUnloadRequest { model_id: Some(orch.model_id.clone()) }),
                    )
                    .await
                    .ok();
                    orchestrator_unloaded = Some(orch);
                    loaded = load_model(
                        State(state.clone()),
                        Json(ModelLoadRequest {
                            model_path: found.path.clone(),
                            gpu_layers: None,
                            context_size: None,
                            mmproj_path: found.mmproj_path.clone(),
                        }),
                    )
                    .await;
                }
            }
        }

        match loaded {
            Ok(Json(l)) => entry = Some(l),
            Err((_, e)) => {
                if let Some(orch) = &orchestrator_unloaded {
                    reload_orchestrator(state, orch).await;
                }
                return err(tool_call, format!("Failed to load '{}': {}", found.name, e));
            }
        }
    }

    let entry = entry.expect("resolved above");
    let backend_category = crate::task_tags::backend_category_for_tags(&entry.task_tags);
    info!("run_model: '{}' ({:?}) <- {:?}", entry.model_path, entry.task_tags, prompt);

    let result = match backend_category {
        None => {
            if prompt.is_empty() {
                err(tool_call, "run_model needs a 'prompt' for text/vision models.".into())
            } else {
                run_text_model(state, tool_call, entry.port, prompt, params).await
            }
        }
        Some(backend_modality) => {
            let image = match arg(tool_call, "image") {
                Some(handle) => match resolve_media_handle(state, handle).await {
                    Ok(bytes) => Some(bytes),
                    Err(e) => {
                        if let Some(orch) = &orchestrator_unloaded {
                            reload_orchestrator(state, orch).await;
                        }
                        return err(tool_call, e);
                    }
                },
                // The orchestrator often omits the 'image' argument even when the user just
                // attached or generated one earlier in this conversation (e.g. "make a 3d
                // model out of this", or "edit it"). Fall back to the most recent image
                // asset in this conversation for any media-consuming modality.
                None => {
                    match crate::media_store::latest_asset_of_modality(
                        &state.generated_assets, conversation_id, "image",
                    ).await {
                        Some(asset) => match resolve_media_handle(state, &asset.id).await {
                            Ok(bytes) => Some(bytes),
                            Err(_) => None,
                        },
                        None => None,
                    }
                }
            };
            run_media_model(state, tool_call, backend_modality, prompt, job_id, image, params, &entry.model_id, target, conversation_id).await
        }
    };

    if let Some(orch) = &orchestrator_unloaded {
        reload_orchestrator(state, orch).await;
    }
    result
}

/// The currently-loaded text/vision model, if any — the same "plain text model,
/// no media backend_category" predicate `first_chat_port` uses to pick the
/// orchestrator's port.
async fn find_orchestrator_entry(state: &AppState) -> Option<crate::http::LoadedModelEntry> {
    let loaded = state.loaded_models.lock().await;
    loaded
        .values()
        .find(|e| crate::task_tags::backend_category_for_tags(&e.task_tags).is_none())
        .cloned()
}

/// Reload the orchestrator model that was unloaded to make VRAM room for a
/// media/on-demand load. Must be awaited inline before `run_model` returns —
/// the next chat-completion request goes straight to the orchestrator's port.
async fn reload_orchestrator(state: &AppState, orch: &crate::http::LoadedModelEntry) {
    let reloaded = load_model(
        State(state.clone()),
        Json(ModelLoadRequest {
            model_path: orch.model_path.clone(),
            gpu_layers: Some(orch.gpu_layers),
            context_size: Some(orch.context_size),
            mmproj_path: None,
        }),
    )
    .await;
    if let Err((_, e)) = reloaded {
        tracing::warn!("run_model: failed to reload orchestrator '{}': {}", orch.model_path, e);
    }
}

/// Extract the id from a media handle (either `/v1/media/<id>` or a bare id)
/// and look up its bytes in the media store.
async fn resolve_media_handle(state: &AppState, handle: &str) -> Result<Vec<u8>, String> {
    let id = handle.rsplit('/').next().unwrap_or(handle);
    match state.media_store.get(id).await {
        Some((bytes, _mime)) => Ok(bytes),
        None => Err(format!(
            "The image handle '{}' is not available (it may have expired). Generate it again first.",
            handle
        )),
    }
}

async fn find_loaded(state: &AppState, target: &str) -> Option<crate::http::LoadedModelEntry> {
    let loaded = state.loaded_models.lock().await;
    loaded
        .values()
        .find(|l| l.model_id == target || paths_match(&l.model_path, target))
        .cloned()
}

/// Find `target` in the on-disk catalog (used to load-on-demand in `run_model`, and to
/// resolve a modality without loading in `inspect_model`).
async fn find_in_catalog(target: &str) -> Option<crate::http::DetectedModelEntry> {
    let catalog = list_detected_models().await.0;
    catalog
        .into_iter()
        .find(|c| c.name.eq_ignore_ascii_case(target) || paths_match(&c.path, target))
}

/// A hand-written description of the parameters `run_model` accepts for a given modality,
/// returned to the orchestrator by `inspect_model`.
async fn inspect_model(state: &AppState, tool_call: &ToolCall) -> ToolResult {
    let Some(target) = arg(tool_call, "model") else {
        return err(tool_call, "inspect_model needs a 'model' argument.".into());
    };

    let task_tags = match find_loaded(state, target).await {
        Some(entry) => entry.task_tags,
        None => match find_in_catalog(target).await {
            Some(entry) => entry.task_tags,
            None => {
                return err(tool_call, format!(
                    "No model named '{}'. Call list_models to see what is available.", target
                ));
            }
        },
    };
    let backend_category = crate::task_tags::backend_category_for_tags(&task_tags);

    let content = match backend_category {
        None => serde_json::json!([
            {"name": "temperature", "type": "float", "default": 0.7},
            {"name": "top_p", "type": "float", "default": 0.9},
            {"name": "top_k", "type": "int", "default": 40},
            {"name": "max_tokens", "type": "int", "default": 512},
            {"name": "stop_sequences", "type": "array[string]", "default": []}
        ]),
        Some(Modality::Image) => serde_json::json!([
            {"name": "negative_prompt", "type": "string", "optional": true, "note": "backend default if omitted"},
            {"name": "steps", "type": "int", "optional": true, "note": "backend default if omitted"},
            {"name": "cfg_scale", "type": "float", "optional": true, "note": "backend default if omitted"},
            {"name": "width", "type": "int", "optional": true, "note": "backend default if omitted"},
            {"name": "height", "type": "int", "optional": true, "note": "backend default if omitted"},
            {"name": "seed", "type": "int", "optional": true, "note": "backend default if omitted"},
            {"name": "image", "type": "string", "optional": true, "note": "Optional base64-encoded reference image for image-to-image generation"},
            {"name": "strength", "type": "number", "optional": true, "note": "Denoising strength 0-1 for image-to-image, default 0.75; ignored for pure text-to-image"}
        ]),
        Some(Modality::AudioTts) => serde_json::json!([
            {"name": "speed", "type": "float", "optional": true, "note": "backend default if omitted"}
        ]),
        Some(Modality::AudioAsr) => {
            return ok(tool_call, "This modality (speech-to-text) takes no extra parameters.".into());
        }
        Some(backend_modality @ (Modality::Video | Modality::Mesh3D)) => {
            let schema = match state.registry.get_backend(backend_modality).await {
                Some(backend_arc) => backend_arc.read().await.get_param_schema().await,
                None => None,
            };
            match schema {
                Some(v) => v,
                None if backend_modality == Modality::Mesh3D => serde_json::json!({
                    "note": "No detailed schema available; generic fields:",
                    "fields": ["input_kind", "steps", "guidance_scale", "seed", "output_format", "texture", "foreground_ratio"]
                }),
                None => serde_json::json!({
                    "note": "No detailed schema available; generic fields:",
                    "fields": ["negative_prompt", "num_frames", "height", "width", "num_inference_steps", "guidance_scale", "fps", "seed", "output_format"]
                }),
            }
        }
        Some(other) => return err(tool_call, format!("Modality '{:?}' cannot be run yet.", other)),
    };

    ok(tool_call, serde_json::to_string(&content).unwrap_or_else(|_| "[]".into()))
}

/// Text and vision models are served by llama-server on their own port.
async fn run_text_model(
    _state: &AppState,
    tool_call: &ToolCall,
    port: u16,
    prompt: &str,
    params: Option<&serde_json::Value>,
) -> ToolResult {
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let mut body = serde_json::json!({
        "model": "local",
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": 1024,
        "stream": false,
    });
    if let Some(obj) = params.and_then(|p| p.as_object()) {
        for key in ["temperature", "top_p", "top_k", "max_tokens"] {
            if let Some(v) = obj.get(key) {
                body[key] = v.clone();
            }
        }
    }

    let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
    let response = match client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => return err(tool_call, format!("Could not reach the model on port {}: {}", port, e)),
    };
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return err(tool_call, format!("Model returned HTTP {}: {}", status, text));
    }

    let json: serde_json::Value = match response.json().await {
        Ok(j) => j,
        Err(e) => return err(tool_call, format!("Malformed response from the model: {}", e)),
    };
    let text = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string();

    if text.is_empty() {
        return err(tool_call, "The model returned an empty response.".into());
    }
    ok(tool_call, text)
}

/// Everything else goes through its registered backend and may return binary output.
async fn run_media_model(
    state: &AppState,
    tool_call: &ToolCall,
    backend_modality: Modality,
    prompt: &str,
    job_id: &str,
    image: Option<Vec<u8>>,
    params: Option<&serde_json::Value>,
    model_id: &str,
    catalog_name: &str,
    conversation_id: &str,
) -> ToolResult {
    let (mime, modality_static): (&str, &'static str) = match backend_modality {
        Modality::AudioTts => ("audio/wav", "tts"),
        Modality::AudioAsr => ("text/plain", "audio"),
        Modality::Image => ("image/png", "image"),
        Modality::Video => ("video/mp4", "video"),
        Modality::Mesh3D => ("model/gltf-binary", "mesh"),
        other => return err(tool_call, format!("Modality '{:?}' cannot be run yet.", other)),
    };

    let Some(backend_arc) = state.registry.get_backend(backend_modality).await else {
        return err(tool_call, format!("No backend is registered for {:?}.", backend_modality));
    };

    let image_params = if backend_modality == Modality::Image {
        match params {
            Some(p) => match serde_json::from_value::<backend_trait::ImageParams>(p.clone()) {
                Ok(v) => Some(v),
                Err(e) => return err(tool_call, format!("Invalid params for image model: {}", e)),
            },
            None => None,
        }
    } else {
        None
    };

    let audio_params = if backend_modality == Modality::AudioTts {
        match params {
            Some(p) => match serde_json::from_value::<backend_trait::AudioParams>(p.clone()) {
                Ok(v) => Some(v),
                Err(e) => return err(tool_call, format!("Invalid params for tts model: {}", e)),
            },
            None => None,
        }
    } else {
        None
    };

    let mesh_params = if backend_modality == Modality::Mesh3D {
        let mut base = serde_json::json!({ "images": image.clone().map(|b| vec![b]) });
        if let Some(obj) = params.and_then(|p| p.as_object()) {
            const KNOWN: &[&str] = &[
                "input_kind", "steps", "guidance_scale", "seed", "output_format", "texture",
                "foreground_ratio",
            ];
            let mut extra = serde_json::Map::new();
            for (k, v) in obj {
                if KNOWN.contains(&k.as_str()) {
                    base[k] = v.clone();
                } else {
                    extra.insert(k.clone(), v.clone());
                }
            }
            if !extra.is_empty() {
                base["extra"] = serde_json::Value::Object(extra);
            }
        }
        match serde_json::from_value::<backend_trait::Mesh3dParams>(base) {
            Ok(v) => Some(v),
            Err(e) => return err(tool_call, format!("Invalid params for 3D model: {}", e)),
        }
    } else {
        None
    };

    let video_params = if backend_modality == Modality::Video {
        let mut base = serde_json::json!({ "image": image.clone() });
        if let Some(obj) = params.and_then(|p| p.as_object()) {
            const KNOWN: &[&str] = &[
                "negative_prompt", "num_frames", "height", "width", "num_inference_steps",
                "guidance_scale", "fps", "seed", "output_format",
            ];
            let mut extra = serde_json::Map::new();
            for (k, v) in obj {
                if KNOWN.contains(&k.as_str()) {
                    base[k] = v.clone();
                } else {
                    extra.insert(k.clone(), v.clone());
                }
            }
            if !extra.is_empty() {
                base["extra"] = serde_json::Value::Object(extra);
            }
        }
        match serde_json::from_value::<backend_trait::VideoParams>(base) {
            Ok(v) => Some(v),
            Err(e) => return err(tool_call, format!("Invalid params for video model: {}", e)),
        }
    } else {
        None
    };

    let request = InferenceRequest {
        request_id: format!("run_model-{}", uuid::Uuid::new_v4()),
        prompt: prompt.to_string(),
        messages: None,
        sampling: SamplingParams::default(),
        modality: backend_modality,
        image_input: image,
        tools: None,
        tool_choice: None,
        image_params,
        mesh_params,
        audio_params,
        video_params,
        cancel: None,
    };

    crate::http::init_job_progress(state, job_id, modality_static).await;
    let done_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let poller = crate::http::spawn_progress_poller(
        state.job_progress.clone(),
        backend_arc.clone(),
        job_id.to_string(),
        modality_static,
        done_flag.clone(),
    );

    let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    state.job_cancel.lock().await.insert(job_id.to_string(), cancel_flag.clone());
    let mut request = request;
    request.cancel = Some(cancel_flag);

    // run_model waits for generation to finish and hands the result (or the error) straight
    // back in this same tool call - the orchestrator does not need to poll
    // get_generation_progress for the common case. The 15-minute timeout is a generous
    // backstop against a truly hung backend, not a normal-case limit. job_progress is still
    // kept up to date throughout so the UI's independent progress bar keeps working, and
    // get_generation_progress/cancel_job remain available if the orchestrator wants to check
    // on or abort a job out of band (e.g. after starting more than one).
    let backend = backend_arc.read().await;
    let gen_result = tokio::time::timeout(std::time::Duration::from_secs(900), backend.generate(request)).await;
    done_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    poller.abort();
    state.job_cancel.lock().await.remove(job_id);
    drop(backend);

    // Media backends (tts/audio/image/video/mesh3d) are invoked as one-shot orchestrator
    // tools, unlike text models which stay resident across a conversation. Unload here to
    // free VRAM/RAM for whichever model the orchestrator picks next.
    if let Err(e) = backend_arc.write().await.unload_model().await {
        tracing::warn!("Failed to unload {:?} backend after tool call: {}", backend_modality, e);
    }
    // The backend is now unloaded; drop the stale registry entry too, or the next
    // run_model call for this model will find it via find_loaded() and skip re-loading.
    state.loaded_models.lock().await.remove(model_id);

    match gen_result {
        Ok(Ok(response)) => {
            let Some(data) = response.output_data else {
                crate::http::finalize_job_progress(state, job_id, modality_static, None).await;
                return ok(tool_call, response.output_text);
            };
            let media_id = state.media_store.store(data, mime).await;
            crate::media_store::record_generated_asset(
                &state.generated_assets,
                conversation_id,
                crate::media_store::GeneratedAsset {
                    id: media_id.clone(),
                    modality: modality_static.to_string(),
                    model: catalog_name.to_string(),
                    prompt: prompt.to_string(),
                    created_at: std::time::Instant::now(),
                },
            ).await;
            crate::http::finalize_job_progress(state, job_id, modality_static, None).await;
            let media_handle = format!("/v1/media/{}", media_id);
            crate::http::set_job_media(state, job_id, media_handle.clone(), mime.to_string()).await;
            let mut result = ok(tool_call, format!("Generation complete (job_id: {job_id})."));
            result.media_handle = Some(media_handle);
            result.media_type = Some(mime.to_string());
            result
        }
        Ok(Err(e)) => {
            crate::http::finalize_job_progress(state, job_id, modality_static, Some(&e.to_string())).await;
            err(tool_call, format!("The model failed: {}", e))
        }
        Err(_) => {
            crate::http::finalize_job_progress(state, job_id, modality_static, Some("Timed out after 900 seconds.")).await;
            err(tool_call, "The model timed out after 900 seconds.".into())
        }
    }
}
