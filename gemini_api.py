"""
Clean Gemini API wrapper with search functionality and Langfuse observability.
"""
import os
from typing import Optional, Dict
import json
import re
from google import genai
from google.genai import types
from langfuse import observe, get_client

@observe(as_type="generation")
def gemini_create_with_search(
    *, model: str, prompt_name: str, prompt_variables: Optional[Dict] = None
) -> str:
    """
    
    """
    langfuse = get_client()
    prompt = langfuse.get_prompt(prompt_name)
    content_text = prompt.compile(**(prompt_variables or {}))
    langfuse.update_current_generation(
        prompt=prompt,
    )
    
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Create content from prompt
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=content_text)],
        )
    ]

    # Include search and URL context tools
    search_tool = types.Tool(google_search=types.GoogleSearch())
    url_context_tool = types.Tool(url_context=types.UrlContext())
    tools = [search_tool, url_context_tool]

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            tools=tools,
            response_mime_type="text/plain",
        ),
    )

    # Extract response text
    text = response.candidates[0].content.parts[0].text
    
    # Extract grounding metadata
    grounding_metadata = response.candidates[0].grounding_metadata if response.candidates[0].grounding_metadata else None
    
    # Extract sources from grounding_chunks (if available)
    grounding_sources = []
    if grounding_metadata and grounding_metadata.grounding_chunks:
        grounding_sources = [
            {
                "title": chunk.web.title if chunk.web else None,
                "uri": chunk.web.uri if chunk.web else None
            }
            for chunk in grounding_metadata.grounding_chunks
        ]
    
    # Extract search entry point HTML (contains additional source links)
    search_entry_html = None
    if grounding_metadata and grounding_metadata.search_entry_point:
        search_entry_html = grounding_metadata.search_entry_point.rendered_content
    
    # Extract links from search entry HTML
    vertex_links = []
    if search_entry_html:
        # Find all href attributes in anchor tags
        link_pattern = r'<a[^>]+href="([^"]+)"[^>]*>'
        vertex_links = re.findall(link_pattern, search_entry_html)
    
    # Update Langfuse trace with search-related metadata
    search_metadata = {
        "model": model,
        "response_id": response.response_id,
        "model_version": response.model_version,
        "web_search_queries": grounding_metadata.web_search_queries if grounding_metadata else None,
        "grounding_sources_count": len(grounding_sources),
        "grounding_sources": grounding_sources,
        "has_search_entry_point": bool(search_entry_html),
        "vertex_links": vertex_links,
        "vertex_links_count": len(vertex_links),
        "usage_metadata": {
            "total_token_count": response.usage_metadata.total_token_count if response.usage_metadata else None,
            "prompt_token_count": response.usage_metadata.prompt_token_count if response.usage_metadata else None,
            "thoughts_token_count": response.usage_metadata.thoughts_token_count if response.usage_metadata else None,
            "tool_use_prompt_token_count": response.usage_metadata.tool_use_prompt_token_count if response.usage_metadata else None,
        } if response.usage_metadata else None
    }
    
    langfuse.update_current_trace(
        metadata=search_metadata
    )
    
    print(response.model_dump_json(indent=4))
    # Return only the text
    return text