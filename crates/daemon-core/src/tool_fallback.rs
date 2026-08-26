use backend_trait::{ToolCall, ToolSchema};
use uuid::Uuid;

/// Build a system prompt suffix that instructs the model to use tools via JSON blocks.
pub fn build_tool_prompt(tools: &[ToolSchema]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut prompt = String::from(
        "\n\nYou orchestrate the other models on this machine through the following tools. \
         Use them proactively when the user's request would benefit from them — if you are asked \
         for speech, an image, a transcription or a video, run the model that produces it rather \
         than describing what you would do. When you are unsure what is available, call \
         list_models first and pick from what it reports. run_model waits for generation to \
         finish and returns the result directly, so you don't need to poll for it.\n\n"
    );
    for tool in tools {
        prompt.push_str(&format!(
            "### {}\n{}\nParameters: {}\n\n",
            tool.name,
            tool.description,
            serde_json::to_string_pretty(&tool.parameters).unwrap_or_default()
        ));
    }
    prompt.push_str(
        "To use a tool, output a fenced JSON block with the tag `tool_call`:\n\
         ```tool_call\n\
         {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n\
         ```\n\
         Call one tool at a time and wait for its result — the result is handed back to you, \
         and you may then call another tool or answer the user.\n"
    );
    prompt
}

/// Parse the model's output text for tool call fenced blocks.
/// Accepts ```tool_call or ```json fences — many small models use the
/// latter even when instructed to use ```tool_call.
pub fn parse_tool_calls(output: &str) -> Option<ParsedToolCall> {
    for marker_start in &["```tool_call", "```json"] {
        if let Some(parsed) = try_parse_fenced_tool_call(output, marker_start) {
            return Some(parsed);
        }
    }
    try_parse_xml_tool_call(output)
}

/// Qwen-style `<tool_call>` blocks. llama-server only converts these to native
/// `tool_calls` when its grammar trigger fires — a model that emits one while
/// still inside its reasoning block slips past that, so parse it ourselves.
/// Handles both the XML argument form:
///
/// ```text
/// <tool_call>
/// <function=run_model>
/// <parameter=model>
/// model.onnx
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// and a JSON payload wrapped in the same tags.
fn try_parse_xml_tool_call(output: &str) -> Option<ParsedToolCall> {
    let open = "<tool_call>";
    let close = "</tool_call>";
    let start_idx = output.find(open)?;
    let body_start = start_idx + open.len();
    let (body, after_idx) = match output[body_start..].find(close) {
        Some(rel) => (&output[body_start..body_start + rel], body_start + rel + close.len()),
        // An unterminated block still carries a usable call.
        None => (&output[body_start..], output.len()),
    };

    let (name, arguments) = if let Some(parsed) = serde_json::from_str::<serde_json::Value>(body.trim()).ok() {
        let name = parsed.get("name")?.as_str()?.to_string();
        let arguments = parsed.get("arguments").cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        (name, arguments)
    } else {
        let name = between(body, "<function=", ">")?.trim().to_string();
        let mut map = serde_json::Map::new();
        let mut rest = body;
        while let Some(key_start) = rest.find("<parameter=") {
            let after_key = &rest[key_start + "<parameter=".len()..];
            let Some(key_end) = after_key.find('>') else { break };
            let key = after_key[..key_end].trim().to_string();
            let value_region = &after_key[key_end + 1..];
            let value_end = value_region.find("</parameter>").unwrap_or(value_region.len());
            let value = value_region[..value_end].trim().to_string();
            map.insert(key, serde_json::Value::String(value));
            rest = &value_region[value_end.min(value_region.len())..];
        }
        if map.is_empty() {
            return None;
        }
        (name, serde_json::Value::Object(map))
    };

    if name.is_empty() {
        return None;
    }

    Some(ParsedToolCall {
        text_before: output[..start_idx].to_string(),
        tool_call: ToolCall {
            id: format!("xml-{}", Uuid::new_v4()),
            name,
            arguments,
        },
        text_after: output[after_idx..].to_string(),
    })
}

fn between<'a>(haystack: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = haystack.find(open)? + open.len();
    let rest = &haystack[start..];
    let end = rest.find(close)?;
    Some(&rest[..end])
}

fn try_parse_fenced_tool_call(output: &str, marker_start: &str) -> Option<ParsedToolCall> {
    let marker_end = "```";
    let start_idx = output.find(marker_start)?;
    let json_start = start_idx + marker_start.len();
    let remaining = &output[json_start..];
    let end_idx = remaining.find(marker_end)?;
    let json_str = remaining[..end_idx].trim();

    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;

    // Must have "name" to be a tool call (not arbitrary JSON)
    let name = parsed.get("name")?.as_str()?.to_string();
    let arguments = parsed.get("arguments").cloned()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    let tool_call = ToolCall {
        id: format!("fallback-{}", Uuid::new_v4()),
        name,
        arguments,
    };

    Some(ParsedToolCall {
        text_before: output[..start_idx].to_string(),
        tool_call,
        text_after: remaining[end_idx + marker_end.len()..].to_string(),
    })
}

pub struct ParsedToolCall {
    pub text_before: String,
    pub tool_call: ToolCall,
    pub text_after: String,
}

/// Intent-based fallback: if the model's response mentions a registered tool
/// by name but never produced a ```tool_call block, infer the call from
/// context and execute it anyway. This handles small models that understand
/// the intent but can't follow the structured output format.
pub fn detect_tool_intent(
    model_output: &str,
    tools: &[ToolSchema],
    user_prompt: &str,
) -> Option<ToolCall> {
    let lower = model_output.to_lowercase();

    for tool in tools {
        if !lower.contains(&tool.name.to_lowercase()) {
            continue;
        }

        let mut arguments = serde_json::Map::new();

        if let Some(props) = tool.parameters.get("properties").and_then(|p| p.as_object()) {
            for param_name in props.keys() {
                if matches!(param_name.as_str(), "text" | "prompt" | "input" | "description") {
                    let value = extract_quoted_text(user_prompt)
                        .or_else(|| extract_quoted_text(model_output))
                        .unwrap_or_else(|| user_prompt.to_string());
                    arguments.insert(param_name.clone(), serde_json::Value::String(value));
                }
            }
        }

        if !arguments.is_empty() {
            return Some(ToolCall {
                id: format!("intent-{}", Uuid::new_v4()),
                name: tool.name.clone(),
                arguments: serde_json::Value::Object(arguments),
            });
        }
    }

    None
}

/// Last-resort parse for models that emit JSON without any fence or wrapper —
/// a bare `{"name": ..., "arguments": {...}}`, or just an argument object whose
/// keys happen to match one tool's parameters (`{"model": ..., "prompt": ...}`),
/// optionally batched into an array. Only reached after the native, fenced and
/// XML forms have all failed, so a false positive costs one wasted hop.
pub fn parse_loose_json_call(output: &str, tools: &[ToolSchema]) -> Option<ToolCall> {
    let body = output.rsplit("</think>").next().unwrap_or(output);
    for (i, ch) in body.char_indices() {
        if ch != '{' && ch != '[' {
            continue;
        }
        let mut stream =
            serde_json::Deserializer::from_str(&body[i..]).into_iter::<serde_json::Value>();
        if let Some(Ok(value)) = stream.next() {
            if let Some(call) = call_from_json(&value, tools) {
                return Some(call);
            }
        }
    }
    None
}

/// A batch (`[{...}, {...}]`) yields its first usable call — the orchestrator loop
/// runs one tool per hop and the model re-asks for the rest on the next one.
fn call_from_json(value: &serde_json::Value, tools: &[ToolSchema]) -> Option<ToolCall> {
    match value {
        serde_json::Value::Array(items) => items.iter().find_map(|v| call_from_json(v, tools)),
        serde_json::Value::Object(map) => {
            if let Some(name) = map.get("name").and_then(|n| n.as_str()) {
                if tools.iter().any(|t| t.name == name) {
                    return Some(ToolCall {
                        id: format!("loose-{}", Uuid::new_v4()),
                        name: name.to_string(),
                        arguments: map
                            .get("arguments")
                            .cloned()
                            .unwrap_or_else(|| serde_json::Value::Object(Default::default())),
                    });
                }
            }
            // Arguments with the tool name stripped off. Accept only when every key
            // belongs to the same tool and at least two of them match, so a stray
            // one-field JSON snippet in prose can't trigger a call.
            for tool in tools {
                let Some(props) = tool.parameters.get("properties").and_then(|p| p.as_object())
                else {
                    continue;
                };
                if map.len() >= 2 && map.keys().all(|k| props.contains_key(k)) {
                    let mut args = map.clone();
                    // A model improvising this shape also improvises the media handle
                    // ("/v1/generate/cat"), and run_model treats an unresolvable one as
                    // fatal. Drop anything that isn't handle-shaped rather than lose the
                    // whole call to it.
                    args.retain(|key, value| {
                        key != "image" || value.as_str().is_none_or(is_media_handle)
                    });
                    return Some(ToolCall {
                        id: format!("loose-{}", Uuid::new_v4()),
                        name: tool.name.clone(),
                        arguments: serde_json::Value::Object(args),
                    });
                }
            }
            None
        }
        _ => None,
    }
}

/// `run_model` accepts a media handle as `/v1/media/<id>` or a bare id.
fn is_media_handle(value: &str) -> bool {
    match value.strip_prefix("/v1/media/") {
        Some(id) => !id.is_empty(),
        None => !value.is_empty() && !value.contains('/'),
    }
}

/// True when the model wrote as if it had already produced media ("Here are your
/// generated images:", "[An image of a cat]") on a hop where it called no tool at all.
/// Small models do this constantly — `detect_tool_intent` misses it because the text
/// never mentions a tool name. The caller turns a hit into one corrective hop rather
/// than letting the claim reach the user unbacked.
pub fn claims_unbacked_media(output: &str) -> bool {
    const CLAIMS: &[&str] = &[
        "here is the", "here is your", "here is a", "here's the", "here's your", "here's a",
        "here are the", "here are your", "i've generated", "i have generated", "i generated",
        "i've created", "i have created", "i created", "i've made", "i have made", "i made",
        "i've attached", "below is", "below are",
    ];
    const MEDIA: &[&str] = &[
        "image", "picture", "photo", "illustration", "render", "video", "clip", "audio",
        "sound", "speech", "voice", "song", "mesh", "3d model",
    ];

    let lower = output.to_lowercase();
    // Planning to generate inside a reasoning block is fine; only the reply counts.
    let body = lower.rsplit("</think>").next().unwrap_or(&lower);

    body.split(['.', '\n', ':', '!']).any(|sentence| {
        let has_media = MEDIA.iter().any(|m| sentence.contains(m));
        if !has_media {
            return false;
        }
        // A bracketed stand-in ("[an image of a cat]") is a fake on its own.
        CLAIMS.iter().any(|c| sentence.contains(c))
            || sentence.trim_start().starts_with('[')
    })
}

fn extract_quoted_text(text: &str) -> Option<String> {
    let start = text.find('"')?;
    let rest = &text[start + 1..];
    let end = rest.find('"')?;
    let extracted = rest[..end].trim();
    if extracted.is_empty() {
        return None;
    }
    Some(extracted.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen_xml_tool_call_from_reasoning() {
        let output = "<think>\nI should speak it.\n<tool_call>\n<function=run_model>\n\
                      <parameter=model>\nmodel.onnx\n</parameter>\n\
                      <parameter=prompt>\norchestration works\n</parameter>\n\
                      </function>\n</tool_call>\n</think>\n";
        let parsed = parse_tool_calls(output).expect("should parse");
        assert_eq!(parsed.tool_call.name, "run_model");
        assert_eq!(parsed.tool_call.arguments["model"], "model.onnx");
        assert_eq!(parsed.tool_call.arguments["prompt"], "orchestration works");
    }

    #[test]
    fn parses_json_payload_inside_tool_call_tags() {
        let output = r#"<tool_call>{"name": "list_models", "arguments": {"modality": "tts"}}</tool_call>"#;
        let parsed = parse_tool_calls(output).expect("should parse");
        assert_eq!(parsed.tool_call.name, "list_models");
        assert_eq!(parsed.tool_call.arguments["modality"], "tts");
    }

    #[test]
    fn fenced_block_still_wins() {
        let output = "```tool_call\n{\"name\": \"run_model\", \"arguments\": {\"text\": \"hi\"}}\n```";
        let parsed = parse_tool_calls(output).expect("should parse");
        assert_eq!(parsed.tool_call.name, "run_model");
    }

    #[test]
    fn parses_unfenced_batched_argument_objects() {
        // Verbatim shape emitted by flux-orchestration run: an array of run_model
        // argument objects, no fence and no name/arguments wrapper.
        let output = r#"I'll create distinct prompts for each.
[
  {"model": "flux-2-klein-4b-Q8_0.gguf", "image": "/v1/generate/cat", "prompt": "A fluffy orange tabby cat"},
  {"model": "flux-2-klein-4b-Q8_0.gguf", "image": "/v1/generate/dog", "prompt": "A golden retriever puppy"}
]
Both images will be generated."#;
        let tools = crate::orchestrator_tools::schemas();
        let call = parse_loose_json_call(output, &tools).expect("should recover a call");
        assert_eq!(call.name, "run_model");
        assert_eq!(call.arguments["model"], "flux-2-klein-4b-Q8_0.gguf");
        assert_eq!(call.arguments["prompt"], "A fluffy orange tabby cat");
        // The invented handle must not survive — run_model would reject it outright.
        assert!(call.arguments.get("image").is_none());
    }

    #[test]
    fn real_media_handles_survive_the_loose_parser() {
        let tools = crate::orchestrator_tools::schemas();
        for handle in ["/v1/media/abc123", "abc123"] {
            let output = format!(
                r#"{{"model": "shap-e.gguf", "image": "{}", "prompt": "a chair"}}"#,
                handle
            );
            let call = parse_loose_json_call(&output, &tools).expect("should parse");
            assert_eq!(call.arguments["image"], handle);
        }
    }

    #[test]
    fn parses_unfenced_name_arguments_object() {
        let tools = crate::orchestrator_tools::schemas();
        let output = r#"{"name": "list_models", "arguments": {"task_tag": "image"}}"#;
        let call = parse_loose_json_call(output, &tools).expect("should parse");
        assert_eq!(call.name, "list_models");
    }

    #[test]
    fn loose_parser_ignores_unrelated_json() {
        let tools = crate::orchestrator_tools::schemas();
        assert!(parse_loose_json_call(r#"Config: {"theme": "dark", "size": 12}"#, &tools).is_none());
        assert!(parse_loose_json_call("No JSON here at all.", &tools).is_none());
    }

    #[test]
    fn detects_faked_media_replies() {
        assert!(claims_unbacked_media(
            "I'll generate two images for you.\n\nHere are your generated images:"
        ));
        assert!(claims_unbacked_media(
            "[An image of a cute fluffy orange tabby cat sitting on a windowsill]"
        ));
        assert!(claims_unbacked_media("I've created the audio clip you asked for."));
    }

    #[test]
    fn planning_and_questions_are_not_faked_media() {
        assert!(!claims_unbacked_media(
            "Would you like me to generate these images now, or prefer different prompts?"
        ));
        assert!(!claims_unbacked_media("Here is the answer to your question."));
        assert!(!claims_unbacked_media(
            "<think>Here is the image I should make.</think>\n\nWhich style do you want?"
        ));
    }

    #[test]
    fn plain_prose_is_not_a_tool_call() {
        assert!(parse_tool_calls("I will use run_model to do that.").is_none());
    }
}
