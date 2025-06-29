"""Tavily Search API integration tools."""
import logging
import os
from typing import Dict, Optional, Any

import httpx
from dotenv import load_dotenv

from .base import BaseToolError

load_dotenv()

logger = logging.getLogger(__name__)


class TavilySearchError(BaseToolError):
    """Custom exception for Tavily Search API errors."""


class TavilySearchTools:
    """Tools for Tavily Search API integration."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tavily Search client.
        
        Args:
            api_key: Tavily Search API key. If not provided, will use TAVILY_SEARCH_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TAVILY_SEARCH_API_KEY")
        if not self.api_key:
            raise TavilySearchError("TAVILY_SEARCH_API_KEY environment variable is required")

        self.base_url = "https://api.tavily.com"

    async def search(self, query: str) -> Dict[str, Any]:
        """Search the web using Tavily Search API.

        Args:
            query: The search query

        Returns:
            Dictionary containing search results

        Raises:
            TavilySearchError: If search fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": 20,
                        "search_depth": 'advanced',
                        "include_answer": False,
                        "include_images": False,
                        "include_raw_content": False
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    raise TavilySearchError(f"Tavily API returned status {response.status_code}: {response.text}")
                
                data = response.json()
                return {
                    "query": query,
                    "answer": data.get("answer", ""),
                    "results": data.get("results", []),
                    "search_time": data.get("response_time", 0)
                }

        except httpx.RequestError as e:
            raise TavilySearchError(f"Network error during search: {str(e)}") from e
        except Exception as e:
            if isinstance(e, TavilySearchError):
                raise
            raise TavilySearchError(f"Failed to search: {str(e)}") from e


# Global instance for easy access
_SEARCH_TOOLS = None


def get_search_tools() -> TavilySearchTools:
    """Get singleton instance of TavilySearchTools."""
    global _SEARCH_TOOLS
    if _SEARCH_TOOLS is None:
        _SEARCH_TOOLS = TavilySearchTools()
    return _SEARCH_TOOLS


# Tavily Search tool functions for Gemini integration
async def web_search(query: str) -> Dict[str, Any]:
    """Search the web using Tavily Search API.

    Args:
        query: The search query

    Returns:
        Dictionary containing search results and answer
    """
    # Truncate long queries for logging
    query_preview = query[:100] + "..." if len(query) > 100 else query
    logger.info(f"🔍 Web search for: {query_preview}")
    
    try:
        result = await get_search_tools().search(query)
        results_count = len(result.get('results', []))
        search_time = result.get('search_time', 0)
        logger.info(f"✅ Found {results_count} results in {search_time}s")
        return result
    except Exception as e:
        logger.error(f"❌ Web search failed: {str(e)[:200]}")
        raise