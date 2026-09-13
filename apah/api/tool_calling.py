"""Tool / function calling parser and prompt formatting module for Workbench agent workflows."""

import json
import logging
import re
import uuid
from typing import List, Optional, Tuple

from apah.api.schemas import FunctionCall, ToolCall, ToolDefinition

logger = logging.getLogger("apah.api.tool_calling")


TOOL_SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant with access to external tools.
You can call functions to fulfill user requests.

When you need to call a function, respond with a JSON object in the following format:
```json
{
  "name": "function_name",
  "arguments": {
    "arg1": "value1"
  }
}
```

Available tools:
"""


def format_tool_instructions(tools: List[ToolDefinition]) -> str:
    """Format tools into system prompt instructions for structured tool-calling."""
    if not tools:
        return ""

    tool_descs = []
    for tool in tools:
        fn = tool.function
        desc = {
            "name": fn.name,
            "description": fn.description or "",
            "parameters": fn.parameters or {},
        }
        tool_descs.append(json.dumps(desc, indent=2))

    return TOOL_SYSTEM_PROMPT_TEMPLATE + "\n".join(tool_descs) + "\n\nIf no tool call is needed, respond directly with text."


def extract_tool_calls(text: str) -> Tuple[Optional[str], Optional[List[ToolCall]]]:
    """Parse text output for tool calls. Return tuple (content_text, list_of_tool_calls)."""
    if not text:
        return text, None

    # Regex search for ```json ... ``` or raw json object containing "name" and "arguments"
    json_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    candidate_json_str = None

    if json_block_match:
        candidate_json_str = json_block_match.group(1)
    else:
        # Check if entire output or substring looks like a JSON function call
        brace_match = re.search(r"(\{[\s\S]*\"name\"\s*:\s*\"[^\"]+\"[\s\S]*\"arguments\"\s*:[\s\S]*\})", text)
        if brace_match:
            candidate_json_str = brace_match.group(1)

    if candidate_json_str:
        try:
            data = json.loads(candidate_json_str)
            if isinstance(data, dict) and "name" in data:
                fn_name = data["name"]
                raw_args = data.get("arguments", {})
                args_str = json.dumps(raw_args) if isinstance(raw_args, dict) else str(raw_args)

                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=FunctionCall(name=fn_name, arguments=args_str),
                )
                
                # Strip tool call block from text content
                cleaned_content = text.replace(json_block_match.group(0) if json_block_match else candidate_json_str, "").strip()
                return cleaned_content if cleaned_content else None, [tool_call]
        except Exception as e:
            logger.debug(f"Failed to parse candidate tool call JSON: {e}")

    return text, None
