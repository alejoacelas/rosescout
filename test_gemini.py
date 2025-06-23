"""
Test script for Gemini API with JSON output and grounding metadata.
"""
import json
import re
from gemini_api import gemini_create_with_search


def test_gemini_json_output():
    """Test Gemini API with JSON output and grounding metadata."""
    print("🧪 Testing Gemini API with JSON output and grounding")
    print("=" * 50)
    
    try:
        # Call Gemini API
        print("📡 Calling Gemini API...")
        response = gemini_create_with_search(
            model="gemini-2.5-flash",
            prompt_name='pombo-test-json-search',
            prompt_variables=dict(person="Rafael Pombo")
        )
        
        # Extract text and grounding metadata
        raw_text = response
        
        # Extract JSON from response using find/rfind trick
        print("\n🔍 Extracting JSON from response...")
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        
        if start != -1 and end > start:
            json_part = raw_text[start:end]
            try:
                json_data = json.loads(json_part)
                print("✅ JSON extracted and parsed successfully!")
                print(f"\n📄 Parsed JSON:")
                print(json.dumps(json_data, indent=2))
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"🔤 Extracted text: {json_part}")
        else:
            print("❌ No JSON found in response")
                    
    except Exception as e:
        print(f"❌ API call failed: {e}")
        
    print("\n🏁 Test completed!")


if __name__ == "__main__":
    test_gemini_json_output()