#!/usr/bin/env python3
"""
Test script for Google Maps tools and their integration with Gemini.
Updated to use the new generate_content method with direct prompts.
"""
import asyncio
import os
import logging
from dotenv import load_dotenv

from tools import get_coordinates, calculate_distance, web_search, screening_list_search, GoogleMapsError, TavilySearchError, ConsolidatedScreeningListError
from gemini_api import GeminiClient, GeminiAPIError

load_dotenv()

# Configure logging to see tool usage
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_search_tool_independently():
    """Test the Tavily search tool independently without Gemini integration."""
    print("=" * 60)
    print("TESTING TAVILY SEARCH TOOL INDEPENDENTLY")
    print("=" * 60)
    
    test_queries = [
        "What is the weather like today?",
        "Latest news about artificial intelligence",
        "Python programming tutorial"
    ]
    
    for query in test_queries:
        try:
            result = await web_search(query)
            print(f"✅ Query: {query}")
            print(f"   Answer: {result['answer'][:100]}..." if result['answer'] else "   No direct answer")
            print(f"   Results: {len(result['results'])} found")
            if result['results']:
                print(f"   First result: {result['results'][0]['title']}")
        except TavilySearchError as e:
            print(f"❌ Query: {query}")
            print(f"   Error: {e}")
        print()


async def test_screening_tool_independently():
    """Test the Consolidated Screening List tool independently without Gemini integration."""
    print("=" * 60)
    print("TESTING CONSOLIDATED SCREENING LIST TOOL INDEPENDENTLY")
    print("=" * 60)
    
    test_searches = [
        {"name": "Sberbank", "description": "Search by company name"},
    ]
    
    for search in test_searches:
        try:
            result = await screening_list_search(**{k: v for k, v in search.items() if k != "description"})
            print(f"✅ {search['description']}")
            print(f"   Total results: {result.get('total_returned', 0)}")
            if result.get('results'):
                first_result = result['results'][0]
                print(f"   First result: {first_result.get('name', 'N/A')}")
                if first_result.get('addresses'):
                    addr = first_result['addresses'][0]
                    print(f"   Location: {addr.get('city', 'N/A')}, {addr.get('country', 'N/A')}")
        except ConsolidatedScreeningListError as e:
            print(f"❌ {search['description']}")
            print(f"   Error: {e}")
        print()


def test_maps_tools_independently():
    """Test the Maps tools independently without Gemini integration."""
    print("=" * 60)
    print("TESTING GOOGLE MAPS TOOLS INDEPENDENTLY")
    print("=" * 60)
    
    # Test get_coordinates
    print("\n1. Testing get_coordinates function:")
    test_addresses = [
        "1600 Amphitheatre Parkway, Mountain View, CA",
        "Times Square, New York, NY",
        "invalid address that should not exist 12345"
    ]
    
    for address in test_addresses:
        try:
            result = get_coordinates(address)
            print(f"✅ Address: {address}")
            print(f"   Coordinates: {result['latitude']}, {result['longitude']}")
            print(f"   Formatted: {result['formatted_address']}")
        except GoogleMapsError as e:
            print(f"❌ Address: {address}")
            print(f"   Error: {e}")
        print()
    
    # Test calculate_distance
    print("\n2. Testing calculate_distance function:")
    test_routes = [
        ("San Francisco, CA", "Los Angeles, CA"),
        ("New York, NY", "Boston, MA"),
        ("invalid address 1", "invalid address 2")
    ]
    
    for origin, destination in test_routes:
        try:
            result = calculate_distance(origin, destination)
            print(f"✅ Route: {origin} → {destination}")
            print(f"   Distance: {result['distance_km']:.2f} km ({result['distance_text']})")
            print(f"   Duration: {result['duration']}")
        except GoogleMapsError as e:
            print(f"❌ Route: {origin} → {destination}")
            print(f"   Error: {e}")
        print()

async def test_gemini_with_maps_tools():
    """Test Gemini integration with Maps tools using direct prompts."""
    print("=" * 60)
    print("TESTING GEMINI WITH MAPS TOOLS (DIRECT PROMPTS)")
    print("=" * 60)
    
    try:
        client = GeminiClient()
        
        # Test scenarios with direct prompts
        test_cases = [
            {
                "prompt": "What are the coordinates of the Eiffel Tower in Paris, France?",
                "tools": [get_coordinates],
                "description": "Get coordinates using Maps tool"
            },
            {
                "prompt": "How far is it from New York, NY to Boston, MA in kilometers?",
                "tools": [calculate_distance],
                "description": "Calculate distance using Maps tool"
            },
            {
                "prompt": "Find the coordinates of 1600 Amphitheatre Parkway, Mountain View, CA and then calculate the distance from there to San Francisco, CA in kilometers.",
                "tools": [get_coordinates, calculate_distance],
                "description": "Use both Maps tools"
            },
            {
                "prompt": "What is the weather like today?",
                "tools": [web_search],
                "description": "Search the web using Tavily"
            },
            {
                "prompt": "What is the weather like today and find the distance between London and Paris?",
                "tools": [web_search, calculate_distance],
                "description": "Combine Tavily search with Maps tools"
            },
            {
                "prompt": "Search for companies named 'Huawei' in the consolidated screening list",
                "tools": [screening_list_search],
                "description": "Search screening list by company name"
            },
            {
                "prompt": "Find companies in Beijing on the screening list and calculate distance from Beijing to Shanghai",
                "tools": [screening_list_search, calculate_distance],
                "description": "Combine screening list search with Maps tools"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['description']}")
            print(f"   Prompt: '{test_case['prompt']}'")
            
            try:
                response = await client.generate_content(
                    model="gemini-2.5-flash",
                    prompt=test_case["prompt"],
                    tools=test_case["tools"]
                )
                print(f"✅ Response: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
            print()
    
    except GeminiAPIError as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def test_gemini_without_tools():
    """Test Gemini without any tools for comparison."""
    print("=" * 60)
    print("TESTING GEMINI WITHOUT TOOLS")
    print("=" * 60)
    
    try:
        client = GeminiClient()
        
        test_prompts = [
            "What are the coordinates of the Eiffel Tower?",
            "How far is it from New York to Boston?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Testing without tools: '{prompt}'")
            try:
                response = await client.generate_content(
                    model="gemini-2.5-flash",
                    prompt=prompt
                )
                print(f"✅ Response: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
            print()
    
    except Exception as e:
        print(f"❌ Failed to test without tools: {e}")


def check_environment():
    """Check if required environment variables are set."""
    print("=" * 60)
    print("CHECKING ENVIRONMENT SETUP")
    print("=" * 60)
    
    required_vars = ["GOOGLE_MAPS_API_KEY", "GEMINI_API_KEY", "TAVILY_SEARCH_API_KEY", "CONSOLIDATED_SCREENING_LIST_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is NOT set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return False
    
    print("\n✅ All required environment variables are set")
    return True

async def main():
    """Run all tests."""
    print("Google Maps Tools Test Suite (Updated)")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        return
    
    # Test search tool independently
    await test_search_tool_independently()
    
    # Test screening tool independently
    await test_screening_tool_independently()
    
    # Test maps tools independently
    test_maps_tools_independently()
    
    # Test Gemini with tools
    await test_gemini_with_maps_tools()
    
    # Test Gemini without tools for comparison
    await test_gemini_without_tools()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())