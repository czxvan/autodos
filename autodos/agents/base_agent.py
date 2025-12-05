"""Base agent class."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel
from json_repair import loads as json_loads

from autodos.config import AgentConfig
from anyllm import AsyncClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base agent.
        
        Args:
            config: Agent configuration dictionary
        """
        self.config = AgentConfig(**config)
        
        # Initialize anyllm async client
        client_kwargs = {
            'model': self.config.model,
            'backend': self.config.backend_type,
        }
        
        # Add optional configuration
        if self.config.api_key:
            client_kwargs['api_key'] = self.config.api_key
        if self.config.base_url:
            client_kwargs['base_url'] = self.config.base_url
        if self.config.provider:
            client_kwargs['provider'] = self.config.provider
        
        self.client = AsyncClient(**client_kwargs)
        self.system_prompt = self._get_system_prompt()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get system prompt for this agent.
        
        Returns:
            System prompt string
        """
        pass
    
    @property
    @abstractmethod
    def response_model(self) -> Optional[Type[BaseModel]]:
        """Get response model for structured output.
        
        Returns:
            Pydantic model class or None
        """
        return None
    
    def _get_json_schema_prompt(self) -> str:
        """Generate JSON schema prompt from response model.
        
        Returns:
            JSON schema description string
        """
        if not self.response_model:
            return ""
        
        # Get Pydantic model schema
        schema = self.response_model.model_json_schema()
        
        # Build readable schema description
        lines = ["\n\nYou MUST respond with ONLY a valid JSON object in the following format:"]
        lines.append("{")
        
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        for field_name, field_info in properties.items():
            field_type = field_info.get('type', 'string')
            field_desc = field_info.get('description', '')
            is_required = field_name in required
            
            # Handle array types
            if field_type == 'array':
                items_type = field_info.get('items', {}).get('type', 'string')
                type_str = f"array of {items_type}"
            else:
                type_str = field_type
            
            req_mark = "(required)" if is_required else "(optional)"
            desc_str = f" // {field_desc}" if field_desc else ""
            lines.append(f'  "{field_name}": <{type_str}> {req_mark}{desc_str}')
        
        lines.append("}")
        lines.append("\nDo NOT include any text before or after the JSON object.")
        
        return "\n".join(lines)
    
    def _create_messages(
        self,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Create message list for LLM request.
        
        Args:
            user_prompt: User prompt string
            history: Optional conversation history (list of dicts with 'role' and 'content')
            system_prompt: Optional custom system prompt
            
        Returns:
            List of messages as dicts
        """
        messages = []
        
        # Add system prompt with JSON schema if needed
        system = system_prompt or self.system_prompt
        
        if self.response_model and system:
            # Add JSON schema to system prompt
            json_schema = self._get_json_schema_prompt()
            system = system + json_schema
        
        if system:
            messages.append({"role": "system", "content": system})
        
        # Add history
        if history:
            messages.extend(history)
        
        # Add user prompt
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    async def request(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 5,
        **kwargs,
    ) -> Any:
        """Send async request to LLM with retry logic.
        
        Args:
            prompt: User prompt
            history: Optional conversation history
            max_retries: Maximum number of retry attempts
            **kwargs: Additional arguments for LLM client
            
        Returns:
            Response object processed by to_result()
        """
        import asyncio
        from anyllm import to_result
        
        messages = self._create_messages(prompt, history)
        
        # Add temperature and max_tokens from config if not in kwargs
        if 'temperature' not in kwargs:
            kwargs['temperature'] = self.config.temperature
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = self.config.max_tokens
        
        logger.debug(f"Sending request to {self.config.model}...")
        
        # Retry loop for handling rate limits and transient errors
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}...")
                # Add timeout protection (default 180 seconds)
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        messages=messages,
                        **kwargs,
                    ),
                    timeout=180.0
                )
                
                # Use to_result to get normalized response
                result = to_result(response)
                logger.debug(f"✓ Received response ({len(result['content'])} chars)")
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"✗ Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    wait_time = min((2 ** attempt) * 3, 30)  # Exponential backoff
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"✗ Request timeout after {max_retries} attempts")
                    raise
            except Exception as e:
                error_msg = str(e).lower()
                is_retryable = any(keyword in error_msg for keyword in [
                    'busy', 'rate limit', 'timeout', 'temporary', 'retry', 'overloaded', 'slow',
                    'empty reply', 'curl', 'connection', 'network', 'unreachable', 'failed to perform'
                ])
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = min((2 ** attempt) * 3, 30)  # Exponential backoff: 3s, 6s, 12s, 24s, 30s max
                    logger.warning(f"✗ Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"✗ Request failed after {attempt + 1} attempts: {e}")
                    raise
    
    def parse_response(self, result: Any) -> Any:
        """Parse LLM response into structured format.
        
        Args:
            result: Response result dict from to_result() with 'content', 'usage', 'finish_reason'
            
        Returns:
            Parsed response (Pydantic model or raw result)
        """
        if self.response_model:
            try:
                # Extract content from result dict
                content = result['content'] if isinstance(result, dict) else str(result)
                
                # Use json_repair to handle malformed JSON
                # It can handle markdown code blocks, extra text, incomplete JSON, etc.
                data = json_loads(content)
                return self.response_model(**data)
                
            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
                content = result['content'] if isinstance(result, dict) else str(result)
                logger.error(f"Content preview: {content[:500]}")
                raise
        
        return result
    
    async def get_reply(self, prompt: str, **kwargs) -> Any:
        """Get parsed reply from agent.
        
        Args:
            prompt: User prompt
            **kwargs: Additional arguments
            
        Returns:
            Parsed reply (Pydantic model for structured output, or result object)
        """
        result = await self.request(prompt, **kwargs)
        return self.parse_response(result)
