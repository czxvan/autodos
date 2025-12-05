"""AutoDoS - Automated Denial-of-Service Attack for LLMs

A redesigned and reimplemented framework for LLM DoS attacks with clear architecture
and easy extensibility.
"""

__version__ = "0.1.0"

from autodos.attack import AutoDoSAttack
from autodos.config import AutoDoSConfig, Message, AgentConfig, AttackResult

__all__ = [
    "AutoDoSAttack",
    "AutoDoSConfig",
    "Message",
    "AgentConfig",
    "AttackResult",
]
