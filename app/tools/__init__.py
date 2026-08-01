"""
Tool package. Importing it registers every tool.

Registration happens as an import side effect of the submodules, so this module imports them
deliberately. Anything needing a populated registry should import from here rather than from
``registry`` directly, or it will see an empty registry depending on import order.
"""
from __future__ import annotations

from app.tools import analysis_tools, research_tools  # noqa: F401  (registration side effect)
from app.tools.registry import (
    TOOL_REGISTRY,
    ToolContext,
    ToolError,
    ToolPermissionError,
    ToolSpec,
    ToolValidationError,
    all_specs,
    get_spec,
    openai_tool_schemas,
    permission_matrix,
    register_tool,
    run_tool,
    tools_for,
)

__all__ = [
    "TOOL_REGISTRY", "ToolContext", "ToolError", "ToolPermissionError", "ToolSpec",
    "ToolValidationError", "all_specs", "get_spec", "openai_tool_schemas",
    "permission_matrix", "register_tool", "run_tool", "tools_for",
]
