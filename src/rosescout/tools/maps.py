"""Google Maps API integration tools."""
import logging
import os
from typing import Dict, Optional, Any

import googlemaps
from dotenv import load_dotenv

from .base import BaseToolError

load_dotenv()

logger = logging.getLogger(__name__)


class GoogleMapsError(BaseToolError):
    """Custom exception for Google Maps API errors."""


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

    def calculate_distance(self, origin_address: str, destination_address: str) -> Dict[str, Any]:
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


# Global instance for easy access
_MAPS_TOOLS = None


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
    # Truncate long addresses for logging
    address_preview = address[:100] + "..." if len(address) > 100 else address
    logger.info(f"🗺️  Getting coordinates for: {address_preview}")
    
    try:
        result = get_maps_tools().get_coordinates(address)
        logger.info(f"✅ Coordinates found: {result.get('latitude', 'N/A')}, {result.get('longitude', 'N/A')}")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to get coordinates: {str(e)[:200]}")
        raise


def calculate_distance(origin_address: str, destination_address: str) -> Dict[str, Any]:
    """Calculate distance between two addresses in kilometers.

    Args:
        origin_address: Starting address
        destination_address: Destination address

    Returns:
        Dictionary containing distance information including distance in kilometers
    """
    # Truncate long addresses for logging
    origin_preview = origin_address[:50] + "..." if len(origin_address) > 50 else origin_address
    dest_preview = destination_address[:50] + "..." if len(destination_address) > 50 else destination_address
    logger.info(f"📏 Calculating distance: {origin_preview} → {dest_preview}")
    
    try:
        result = get_maps_tools().calculate_distance(origin_address, destination_address)
        distance_km = result.get('distance_km', 'N/A')
        duration = result.get('duration', 'N/A')
        logger.info(f"✅ Distance calculated: {distance_km} km, duration: {duration}")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to calculate distance: {str(e)[:200]}")
        raise