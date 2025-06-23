"""
Test script for Gemini API with async parallel processing for multiple historical figures.
"""
import json
import re
import asyncio
import time
from gemini_api import gemini_create_with_search_async


async def test_single_historical_figure(person_name: str):
    """Test Gemini API with a single historical figure."""
    print(f"🔍 Processing {person_name}...")
    
    try:
        # Call Gemini API asynchronously
        response = await gemini_create_with_search_async(
            model="gemini-2.5-flash",
            prompt_name='pombo-test-json-search',
            prompt_variables=dict(person=person_name)
        )
        
        # Extract text and grounding metadata
        raw_text = response
        
        # Extract JSON from response using find/rfind trick
        print(f"\n🔍 Extracting JSON from response for {person_name}...")
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        
        if start != -1 and end > start:
            json_part = raw_text[start:end]
            try:
                json_data = json.loads(json_part)
                print(f"✅ JSON extracted and parsed successfully for {person_name}!")
                print(f"\n📄 Parsed JSON for {person_name}:")
                print(json.dumps(json_data, indent=2))
                return {"person": person_name, "success": True, "data": json_data}
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed for {person_name}: {e}")
                print(f"🔤 Extracted text: {json_part}")
                return {"person": person_name, "success": False, "error": f"JSON parsing failed: {e}"}
        else:
            print(f"❌ No JSON found in response for {person_name}")
            return {"person": person_name, "success": False, "error": "No JSON found in response"}
                    
    except Exception as e:
        print(f"❌ API call failed for {person_name}: {e}")
        return {"person": person_name, "success": False, "error": f"API call failed: {e}"}


async def test_gemini_parallel_historical_figures():
    """Test Gemini API with 5 historical figures in parallel."""
    print("🧪 Testing Gemini API with 5 historical figures in parallel")
    print("=" * 70)
    
    # 5 historical figures to test
    historical_figures = [
        "Napoleon Bonaparte",
        "Leonardo da Vinci", 
        "Marie Curie",
        "Albert Einstein",
        "Cleopatra"
    ]
    
    start_time = time.time()
    
    try:
        # Create tasks for all historical figures
        print(f"📡 Starting parallel processing for {len(historical_figures)} historical figures...")
        
        tasks = [test_single_historical_figure(person) for person in historical_figures]
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n⏱️ Total processing time: {total_time:.2f} seconds")
        print(f"📊 Average time per person: {total_time/len(historical_figures):.2f} seconds")
        
        # Process results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result)})
            elif result.get("success", False):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        print(f"\n📈 Results Summary:")
        print(f"✅ Successful: {len(successful_results)}")
        print(f"❌ Failed: {len(failed_results)}")
        
        if failed_results:
            print(f"\n❌ Failed requests:")
            for failed in failed_results:
                print(f"  - {failed.get('person', 'Unknown')}: {failed.get('error', 'Unknown error')}")
        
        return {
            "total_time": total_time,
            "successful_count": len(successful_results),
            "failed_count": len(failed_results),
            "successful_results": successful_results,
            "failed_results": failed_results
        }
        
    except Exception as e:
        print(f"❌ Parallel processing failed: {e}")
        return {"error": str(e)}


async def main():
    """Main async function to run the parallel test."""
    results = await test_gemini_parallel_historical_figures()
    print("\n🏁 Test completed!")
    return results


if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())