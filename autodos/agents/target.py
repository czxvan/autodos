"""Target agent - the agent being attacked."""

from typing import Optional, Type
from pydantic import BaseModel

from autodos.agents.base_agent import BaseAgent


class TargetAgent(BaseAgent):
    """Target agent that will be attacked."""
    
    def __init__(self, config: dict, function_description: str):
        """Initialize target agent.
        
        Args:
            config: Agent configuration
            function_description: Description of target's function
        """
        self.function_description = function_description
        super().__init__(config)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for target agent.
        
        Returns:
            System prompt
        """
        return self.function_description
    
    @property
    def response_model(self) -> Optional[Type[BaseModel]]:
        """Target agent doesn't need structured output.
        
        Returns:
            None
        """
        return None
