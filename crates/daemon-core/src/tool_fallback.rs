use backend_trait::{ToolCall, ToolSchema};
use uuid::Uuid;

/// Build a system prompt suffix that instructs the model to use tools via JSON blocks.
pub fn build_tool_prompt(tools: &[ToolSchema]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut prompt = String::from(
        "\n\nYou orchestrate the other models on this machine through the following tools. \
         Most messages — greetings, general knowledge questions, chit-chat, anything you can \
         just answer directly — need NO tool call at all. Only reach for a tool when the user's \
         request actually requires generating media (speech, an image, a transcription, a video) \
         or otherwise needs information about what is installed. Never call list_models 'just in \
         case' or as a default first step — only call it when you specifically need to check what \
         models are available to fulfill the current request. run_model waits for generation to \
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
    fn plain_prose_is_not_a_tool_call() {
        assert!(parse_tool_calls("I will use run_model to do that.").is_none());
    }
}
