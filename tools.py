"""
Custom tools for Gemini integration with external APIs.
"""
import os
from typing import Dict, Optional

import httpx
import googlemaps
from dotenv import load_dotenv

load_dotenv()

class GoogleMapsError(Exception):
    """Custom exception for Google Maps API errors."""


class TavilySearchError(Exception):
    """Custom exception for Tavily Search API errors."""


class ConsolidatedScreeningListError(Exception):
    """Custom exception for Consolidated Screening List API errors."""

class GoogleMapsTools:
    """Tools for Google Maps API integration."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Google Maps client.
        
        Args:
            api_key: Google Maps API key. If not provided, will use GOOGLE_MAPS_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise GoogleMapsError("GOOGLE_MAPS_API_KEY environment variable is required")

        self.client = googlemaps.Client(key=self.api_key)

    def get_coordinates(self, address: str) -> Dict[str, float]:
        """Get latitude and longitude coordinates for an address.

        Args:
            address: The address to geocode (e.g., "1600 Amphitheatre Parkway, Mountain View, CA")

        Returns:
            Dictionary containing latitude and longitude coordinates

        Raises:
            GoogleMapsError: If geocoding fails or address not found
        """
        try:
            geocode_result = self.client.geocode(address)

            if not geocode_result:
                raise GoogleMapsError(f"No results found for address: {address}")

            location = geocode_result[0]['geometry']['location']
            formatted_address = geocode_result[0]['formatted_address']

            return {
                "latitude": location['lat'],
                "longitude": location['lng'],
                "formatted_address": formatted_address,
                "address": address
            }

        except Exception as e:
            if isinstance(e, GoogleMapsError):
                raise
            raise GoogleMapsError(f"Failed to geocode address '{address}': {str(e)}") from e

    def calculate_distance(self, origin_address: str, destination_address: str) -> Dict[str, any]:
        """Calculate distance between two addresses in kilometers.

        Args:
            origin_address: Starting address
            destination_address: Destination address

        Returns:
            Dictionary containing distance information

        Raises:
            GoogleMapsError: If distance calculation fails
        """
        try:
            distance_result = self.client.distance_matrix(
                origins=[origin_address],
                destinations=[destination_address],
                units="metric"
            )

            if (not distance_result['rows'] or
                not distance_result['rows'][0]['elements'] or
                distance_result['rows'][0]['elements'][0]['status'] != 'OK'):
                msg = f"Could not calculate distance between '{origin_address}' and '{destination_address}'"
                raise GoogleMapsError(msg)

            element = distance_result['rows'][0]['elements'][0]

            return {
                "origin_address": distance_result['origin_addresses'][0],
                "destination_address": distance_result['destination_addresses'][0],
                "distance_km": element['distance']['value'] / 1000,  # Convert meters to kilometers
                "distance_text": element['distance']['text'],
                "duration": element['duration']['text'],
                "duration_seconds": element['duration']['value']
            }

        except Exception as e:
            if isinstance(e, GoogleMapsError):
                raise
            raise GoogleMapsError(f"Failed to calculate distance: {str(e)}") from e


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

    async def search(self, query: str) -> Dict[str, any]:
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
                    city: Optional[str] = None, state: Optional[str] = None) -> Dict[str, any]:
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


# Global instances for easy access
_MAPS_TOOLS = None
_SEARCH_TOOLS = None
_SCREENING_TOOLS = None


def get_maps_tools() -> GoogleMapsTools:
    """Get singleton instance of GoogleMapsTools."""
    global _MAPS_TOOLS
    if _MAPS_TOOLS is None:
        _MAPS_TOOLS = GoogleMapsTools()
    return _MAPS_TOOLS


# Tool functions for Gemini integration
def get_coordinates(address: str) -> Dict[str, float]:
    """Get latitude and longitude coordinates for an address.

    Args:
        address: The address to geocode (e.g., "1600 Amphitheatre Parkway, Mountain View, CA")

    Returns:
        Dictionary containing latitude and longitude coordinates
    """
    return get_maps_tools().get_coordinates(address)


def calculate_distance(origin_address: str, destination_address: str) -> Dict[str, any]:
    """Calculate distance between two addresses in kilometers.

    Args:
        origin_address: Starting address
        destination_address: Destination address

    Returns:
        Dictionary containing distance information including distance in kilometers
    """
    return get_maps_tools().calculate_distance(origin_address, destination_address)


def get_search_tools() -> TavilySearchTools:
    """Get singleton instance of TavilySearchTools."""
    global _SEARCH_TOOLS
    if _SEARCH_TOOLS is None:
        _SEARCH_TOOLS = TavilySearchTools()
    return _SEARCH_TOOLS


def get_screening_tools() -> ConsolidatedScreeningListTools:
    """Get singleton instance of ConsolidatedScreeningListTools."""
    global _SCREENING_TOOLS
    if _SCREENING_TOOLS is None:
        _SCREENING_TOOLS = ConsolidatedScreeningListTools()
    return _SCREENING_TOOLS


# Tavily Search tool functions for Gemini integration
async def web_search(query: str) -> Dict[str, any]:
    """Search the web using Tavily Search API.

    Args:
        query: The search query

    Returns:
        Dictionary containing search results and answer
    """
    return await get_search_tools().search(query)


# Consolidated Screening List tool functions for Gemini integration
async def screening_list_search(name: Optional[str] = None, countries: Optional[str] = None, city: Optional[str] = None, state: Optional[str] = None) -> Dict[str, any]:
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
    return await get_screening_tools().search(name=name, countries=countries, city=city, state=state)
