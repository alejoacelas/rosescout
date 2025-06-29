"""Consolidated Screening List API integration tools."""
import logging
import os
from typing import Dict, Optional, Any

import httpx
from dotenv import load_dotenv

from .base import BaseToolError

load_dotenv()

logger = logging.getLogger(__name__)


class ConsolidatedScreeningListError(BaseToolError):
    """Custom exception for Consolidated Screening List API errors."""


class ConsolidatedScreeningListTools:
    """Tools for Consolidated Screening List API integration."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Consolidated Screening List client.
        
        Args:
            api_key: API key for the screening list service. If not provided, will use CONSOLIDATED_SCREENING_LIST_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("CONSOLIDATED_SCREENING_LIST_API_KEY")
        if not self.api_key:
            raise ConsolidatedScreeningListError("CONSOLIDATED_SCREENING_LIST_API_KEY environment variable is required")

        self.base_url = "https://data.trade.gov/consolidated_screening_list/v1"

    async def search(self, name: Optional[str] = None, countries: Optional[str] = None, 
                    city: Optional[str] = None, state: Optional[str] = None) -> Dict[str, Any]:
        """Search the Consolidated Screening List.

        Args:
            name: Searches against the name and alt_names fields
            countries: Searches by country code (ISO alpha-2)
            city: Searches against the city field
            state: Searches against the state field

        Returns:
            Dictionary containing raw JSON response from the API

        Raises:
            ConsolidatedScreeningListError: If search fails
        """
        try:
            params = {"subscription-key": self.api_key}
            if name:
                params["name"] = name
            if countries:
                params["countries"] = countries
            if city:
                params["city"] = city
            if state:
                params["state"] = state

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/search",
                    params=params,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    raise ConsolidatedScreeningListError(f"API returned status {response.status_code}: {response.text}")
                
                return response.json()

        except httpx.RequestError as e:
            raise ConsolidatedScreeningListError(f"Network error during search: {str(e)}") from e
        except Exception as e:
            if isinstance(e, ConsolidatedScreeningListError):
                raise
            raise ConsolidatedScreeningListError(f"Failed to search screening list: {str(e)}") from e


# Global instance for easy access
_SCREENING_TOOLS = None


def get_screening_tools() -> ConsolidatedScreeningListTools:
    """Get singleton instance of ConsolidatedScreeningListTools."""
    global _SCREENING_TOOLS
    if _SCREENING_TOOLS is None:
        _SCREENING_TOOLS = ConsolidatedScreeningListTools()
    return _SCREENING_TOOLS


# Consolidated Screening List tool functions for Gemini integration
async def screening_list_search(
    name: Optional[str] = None,
    countries: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    ) -> Dict[str, Any]:
    """Search the Consolidated Screening List API for sanctioned entities and individuals.
    
    This function searches the consolidated screening lists from the Departments of Commerce, 
    State, and Treasury. The screening list contains entities that are subject to various 
    export restrictions, sanctions, or other trade controls.

    IMPORTANT SEARCH BEHAVIOR:
    - Name matching uses SUBSTRING search - the provided name will match if it appears 
      anywhere within the entity's name or alternate names in the database
    - For effective searches, use key proper nouns (company names, person names) rather 
      than common words or generic terms
    - Examples of good searches: "Huawei", "Kaspersky", "Rosneft", "Al-Qaeda"
    - Examples of poor searches: "technology", "oil", "group", "company"
    - Multiple results may be returned for partial name matches

    Args:
        name: Company name, person name, or entity name to search for (substring match).
              Focus on distinctive proper nouns for best results.
        countries: ISO alpha-2 country code(s) to filter results (e.g., "CN", "RU", "IR")
        city: City name to filter results 
        state: State/province name to filter results

    Returns:
        Dictionary containing the raw JSON response with search results including:
        - total_returned: Number of results returned
        - results: Array of matching entities with details like name, addresses, 
                  source list (SDN, Entity List, etc.), and reason for listing

    Note: This API consolidates multiple screening lists including:
    - SDN (Specially Designated Nationals)  
    - Entity List (Commerce restrictions)
    - Denied Persons List
    - And other regulatory lists
    """
    # Build search parameters string for logging
    search_params = []
    if name:
        name_preview = name[:50] + "..." if len(name) > 50 else name
        search_params.append(f"name='{name_preview}'")
    if countries:
        search_params.append(f"countries='{countries}'")
    if city:
        search_params.append(f"city='{city}'")
    if state:
        search_params.append(f"state='{state}'")
    
    params_str = ", ".join(search_params) if search_params else "no parameters"
    logger.info(f"🔍 Screening list search with {params_str}")
    
    try:
        result = await get_screening_tools().search(name=name, countries=countries, city=city, state=state)
        total_returned = result.get('total_returned', 0)
        logger.info(f"✅ Found {total_returned} entities in screening lists")
        return result
    except Exception as e:
        logger.error(f"❌ Screening list search failed: {str(e)[:200]}")
        raise