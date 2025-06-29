"""
Async Gemini API wrapper with search functionality and Langfuse observability.
"""
import os
import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Callable, Union

from google import genai
from google.genai import types
from langfuse import observe, get_client

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class GroundingSource:
    """Represents a grounding source from Gemini search results."""
    title: Optional[str] = None
    uri: Optional[str] = None


@dataclass
class UsageMetadata:
    """Token usage metadata from Gemini API response."""
    total_token_count: Optional[int] = None
    prompt_token_count: Optional[int] = None
    thoughts_token_count: Optional[int] = None
    tool_use_prompt_token_count: Optional[int] = None


@dataclass
class SearchMetadata:
    """Comprehensive metadata from Gemini search response."""
    model: str
    response_id: str
    model_version: str
    web_search_queries: Optional[List[str]] = None
    grounding_sources: List[GroundingSource] = None
    vertex_links: List[str] = None
    usage_metadata: Optional[UsageMetadata] = None

    def __post_init__(self):
        if self.grounding_sources is None:
            self.grounding_sources = []
        if self.vertex_links is None:
            self.vertex_links = []


class GeminiAPIError(Exception):
    """Custom exception for Gemini API errors."""
    pass


class GeminiClient:
    """Async Gemini API client with search capabilities."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key. If not provided, will use GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise GeminiAPIError("GEMINI_API_KEY environment variable is required")
        
        self._client = genai.Client(api_key=self.api_key)
        self._langfuse = get_client()
    
    def _extract_grounding_sources(self, grounding_metadata) -> List[GroundingSource]:
        """Extract grounding sources from metadata."""
        if not grounding_metadata or not grounding_metadata.grounding_chunks:
            return []
        
        return [
            GroundingSource(
                title=chunk.web.title if chunk.web else None,
                uri=chunk.web.uri if chunk.web else None
            )
            for chunk in grounding_metadata.grounding_chunks
        ]
    
    def _extract_vertex_links(self, search_entry_html: Optional[str]) -> List[str]:
        """Extract links from search entry HTML."""
        if not search_entry_html:
            return []
        
        link_pattern = r'<a[^>]+href="([^"]+)"[^>]*>'
        return re.findall(link_pattern, search_entry_html)
    
    def _create_usage_metadata(self, usage_metadata) -> Optional[UsageMetadata]:
        """Create UsageMetadata from response."""
        if not usage_metadata:
            return None
        
        return UsageMetadata(
            total_token_count=usage_metadata.total_token_count,
            prompt_token_count=usage_metadata.prompt_token_count,
            thoughts_token_count=usage_metadata.thoughts_token_count,
            tool_use_prompt_token_count=usage_metadata.tool_use_prompt_token_count
        )
    
    def _create_search_metadata(self, response, grounding_sources: List[GroundingSource], 
                               vertex_links: List[str]) -> SearchMetadata:
        """Create comprehensive search metadata."""
        grounding_metadata = response.candidates[0].grounding_metadata
        
        return SearchMetadata(
            model=response.model_version,
            response_id=response.response_id,
            model_version=response.model_version,
            web_search_queries=grounding_metadata.web_search_queries if grounding_metadata else None,
            grounding_sources=grounding_sources,
            vertex_links=vertex_links,
            usage_metadata=self._create_usage_metadata(response.usage_metadata)
        )
    
    def _build_generation_config(self, tools: Optional[List[Union[Callable, types.Tool]]] = None) -> types.GenerateContentConfig:
        """Build the generation configuration with specified tools.
        
        Args:
            tools: List of tools to include (functions or Tool objects)
        """
        tool_list = tools or []
        
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            tools=tool_list,
            response_mime_type="text/plain",
        )
    
    def _create_content(self, text: str) -> List[types.Content]:
        """Create content objects for the API request."""
        return [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=text)],
            )
        ]
    
    def _log_tool_usage(self, response, tools: List[Callable]):
        """Log tool usage details."""
        if not response.candidates or not response.candidates[0].content.parts:
            return
            
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                func_call = part.function_call
                tool_names = [tool.__name__ for tool in tools]
                if func_call.name in tool_names:
                    logger.info(f"🔧 Tool used: {func_call.name}")
                    logger.info(f"   Arguments: {dict(func_call.args)}")
                    
                    # Log response if available and not too long
                    if hasattr(part, 'function_response') and part.function_response:
                        response_str = str(part.function_response.response)
                        if len(response_str) <= 200:
                            logger.info(f"   Response: {response_str}")
                        else:
                            logger.info(f"   Response: {response_str[:197]}...")

    @observe(as_type="generation")
    async def generate_content(
        self, 
        *,
        model: str,
        prompt: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Callable]] = None
    ) -> str:
        """
        Generate content using Gemini with specified tools.
        
        Args:
            model: Gemini model name (e.g., 'gemini-2.5-flash')
            prompt: Direct prompt text (if not using Langfuse)
            prompt_name: Name of the prompt in Langfuse (if not using direct prompt)
            prompt_variables: Variables to substitute in the prompt
            tools: List of tool functions to include
            
        Returns:
            Generated text response
            
        Raises:
            GeminiAPIError: If API call fails or response is invalid
        """
        if not prompt and not prompt_name:
            raise GeminiAPIError("Either prompt or prompt_name must be provided")
        
        try:
            # Create manual test prompt if needed
            if prompt:
                prompt_name = "manual-test-prompt"
                self._langfuse.create_prompt(
                    name=prompt_name,
                    type="text",
                    prompt=prompt, 
                    labels=["production"]
                )
                prompt_identifier = f"{prompt_name}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
            else:
                prompt_identifier = f"{prompt_name}"
                
            # Get and compile prompt
            langfuse_prompt = self._langfuse.get_prompt(prompt_name)
            content_text = langfuse_prompt.compile(**(prompt_variables or {}))
            self._langfuse.update_current_generation(prompt=langfuse_prompt)
            # Log the Gemini call
            logger.info(f"🤖 Gemini call - Model: {model}, Prompt: {prompt_identifier}")
            
            # Build tools list
            tool_list = tools or []
            
            # Log selected tools
            if tool_list:
                tool_names = [tool.__name__ for tool in tool_list]
                logger.info(f"🔧 Selected tools: {', '.join(tool_names)}")
            
            # Create request content and config
            contents = self._create_content(content_text)
            config = self._build_generation_config(tool_list)
            
            # Make async API call
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            
            # Log tool usage
            self._log_tool_usage(response, tool_list or [])
            
            # Validate response
            if not response.candidates or not response.candidates[0].content.parts:
                raise GeminiAPIError("Invalid response from Gemini API")
            
            # Extract response text
            text = response.candidates[0].content.parts[0].text
            
            # Extract and process grounding metadata for Langfuse
            grounding_metadata = response.candidates[0].grounding_metadata
            grounding_sources = self._extract_grounding_sources(grounding_metadata)
            
            search_entry_html = None
            if grounding_metadata and grounding_metadata.search_entry_point:
                search_entry_html = grounding_metadata.search_entry_point.rendered_content
            
            vertex_links = self._extract_vertex_links(search_entry_html)
            
            # Create comprehensive metadata
            search_metadata = self._create_search_metadata(response, grounding_sources, vertex_links)
            
            # Update Langfuse trace with metadata
            self._langfuse.update_current_trace(
                metadata={
                    "model": search_metadata.model,
                    "response_id": search_metadata.response_id,
                    "model_version": search_metadata.model_version,
                    "web_search_queries": search_metadata.web_search_queries,
                    "grounding_sources_count": len(search_metadata.grounding_sources),
                    "grounding_sources": [
                        {"title": src.title, "uri": src.uri} 
                        for src in search_metadata.grounding_sources
                    ],
                    "has_search_entry_point": bool(search_entry_html),
                    "vertex_links": search_metadata.vertex_links,
                    "vertex_links_count": len(search_metadata.vertex_links),
                    "usage_metadata": {
                        "total_token_count": search_metadata.usage_metadata.total_token_count,
                        "prompt_token_count": search_metadata.usage_metadata.prompt_token_count,
                        "thoughts_token_count": search_metadata.usage_metadata.thoughts_token_count,
                        "tool_use_prompt_token_count": search_metadata.usage_metadata.tool_use_prompt_token_count,
                    } if search_metadata.usage_metadata else None
                }
            )
            
            return text
            
        except Exception as e:
            if isinstance(e, GeminiAPIError):
                raise
            raise GeminiAPIError(f"Failed to generate content: {str(e)}") from e
    


