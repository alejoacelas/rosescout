"""
Async Gemini API wrapper with search functionality and Langfuse observability.
"""
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

from google import genai
from google.genai import types
from langfuse import observe, get_client
import langfuse

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
    
    def _build_generation_config(self) -> types.GenerateContentConfig:
        """Build the generation configuration with search tools."""
        search_tool = types.Tool(google_search=types.GoogleSearch())
        url_context_tool = types.Tool(url_context=types.UrlContext())
        
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            tools=[search_tool, url_context_tool],
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
    
    @observe(as_type="generation")
    async def generate_with_search(
        self, 
        *,
        model: str,
        prompt_name: str,
        prompt_variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate content using Gemini with search capabilities.
        
        Args:
            model: Gemini model name (e.g., 'gemini-2.5-flash')
            prompt_name: Name of the prompt in Langfuse
            prompt_variables: Variables to substitute in the prompt
            
        Returns:
            Generated text response
            
        Raises:
            GeminiAPIError: If API call fails or response is invalid
        """
        try:
            # Get and compile prompt from Langfuse
            prompt = self._langfuse.get_prompt(prompt_name)
            content_text = prompt.compile(**(prompt_variables or {}))
            
            # Update Langfuse generation with prompt info
            self._langfuse.update_current_generation(prompt=prompt)
            
            # Create request content and config
            contents = self._create_content(content_text)
            config = self._build_generation_config()
            
            # Make async API call
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            
            # Validate response
            if not response.candidates or not response.candidates[0].content.parts:
                raise GeminiAPIError("Invalid response from Gemini API")
            
            # Extract response text
            text = response.candidates[0].content.parts[0].text
            
            # Extract and process grounding metadata
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
            
            # # Record web search queries count
            # if search_metadata.web_search_queries:
            #     web_search_count = len(search_metadata.web_search_queries)
            #     self._langfuse.score(
            #         trace_id=self._langfuse.get_current_trace_id(),
            #         name="web_search_queries_count",
            #         value=web_search_count,
            #     )
            
            # # Record thoughts token count
            # if search_metadata.usage_metadata and search_metadata.usage_metadata.thoughts_token_count:
            #     self._langfuse.score(
            #         trace_id=self._langfuse.get_current_trace_id(),
            #         name="thoughts_token_count",
            #         value=search_metadata.usage_metadata.thoughts_token_count,
            #     )
            
            return text
            
        except Exception as e:
            if isinstance(e, GeminiAPIError):
                raise
            raise GeminiAPIError(f"Failed to generate content: {str(e)}") from e


