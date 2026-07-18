"""Dynamic construction of agents declared in the central configuration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from agents.base_agent import BaseAgent


def build_agent(specification: dict[str, Any]) -> BaseAgent:
    """Instantiate a configured agent from its dotted class path."""
    class_path = specification["class_path"]
    module_name, separator, class_name = class_path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid agent class path: {class_path}")
    module = import_module(module_name)
    agent_class = getattr(module, class_name)
    agent = agent_class(**specification.get("kwargs", {}))
    if not isinstance(agent, BaseAgent):
        raise TypeError(f"{class_path} is not a BaseAgent implementation")
    return agent
