"""Configuration and type definitions for AutoDoS."""

import os
import re
from typing import Any, Dict, Literal, Optional, TypedDict
from dataclasses import dataclass
import yaml
from pydantic import BaseModel, Field


# ============================================================================
# Message Type
# ============================================================================

class Message(TypedDict):
    """Chat message type."""
    role: Literal["system", "user", "assistant"]
    content: str


# ============================================================================
# Agent Configuration
# ============================================================================

@dataclass
class AgentConfig:
    """Agent configuration."""
    model: str
    backend_type: str = "openai"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    provider: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 1.0


# ============================================================================
# Attack Configuration
# ============================================================================

class AttackParams(BaseModel):
    """Attack parameters."""
    # Tree generation parameters
    n_subproblems: int = Field(default=25, ge=5, le=50)
    question_length: int = Field(default=200, ge=50, le=1000)
    
    # Optimization parameters
    optimize_iterations: int = Field(default=10, ge=1, le=50)
    n_optimization_streams: int = Field(default=3, ge=1, le=10, description="Number of parallel optimization streams")
    
    # Concurrency control
    max_concurrent_requests: int = Field(default=3, ge=1, le=10, description="Maximum concurrent API requests")


class TargetConfig(BaseModel):
    """Target configuration."""
    function_description: str


class AgentsConfig(BaseModel):
    """All agents configuration."""
    target: Dict[str, Any]
    optimize: Dict[str, Any]
    judge: Dict[str, Any]
    deep_backtracking: Dict[str, Any]
    breadth_expansion: Dict[str, Any]


class LoggingConfig(BaseModel):
    """Logging configuration."""
    output_dir: str = "logs"
    save_requests: bool = True
    save_responses: bool = True


class OutputConfig(BaseModel):
    """Output configuration."""
    save_dir: str = "outputs"
    save_intermediate: bool = True


class AutoDoSConfig(BaseModel):
    """Main AutoDoS configuration."""
    attack: AttackParams
    target: TargetConfig
    agents: AgentsConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "AutoDoSConfig":
        """Load from YAML file with environment variable expansion.
        
        Supports ${VAR_NAME} syntax, e.g., api_key: ${OPENAI_API_KEY}
        """
        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**cls._expand_env_vars(config_dict))
    
    @staticmethod
    def _expand_env_vars(obj: Any) -> Any:
        """Recursively replace ${VAR_NAME} with environment variables."""
        if isinstance(obj, dict):
            return {key: AutoDoSConfig._expand_env_vars(val) for key, val in obj.items()}
        
        if isinstance(obj, list):
            return [AutoDoSConfig._expand_env_vars(item) for item in obj]
        
        if isinstance(obj, str):
            # Replace ${VAR_NAME} with os.getenv("VAR_NAME")
            return re.sub(
                r'\$\{([^}]+)\}',
                lambda match: os.getenv(match.group(1), match.group(0)),
                obj
            )
        
        return obj


# ============================================================================
# Result Type
# ============================================================================

class AttackResult(BaseModel):
    """Attack result."""
    success: bool
    prompt: str
    response_content: Optional[str] = None
    response_length: int = 0
    iteration: int = 0
    total_cost: float = 0.0
    error: Optional[str] = None
