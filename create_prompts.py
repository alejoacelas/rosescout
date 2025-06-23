"""Script to create Langfuse prompts for the demo."""
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

# Initialize Langfuse client
langfuse = get_client()

def create_prompts():
    """Create prompts in Langfuse for the demo."""
    
    # 1. User Profile Extraction Prompt
    profile_prompt = langfuse.create_prompt(
        name="user-profile-extraction",
        prompt="Extract user profile information from the given text. Focus on identifying the person's name, age, interests, location, and occupation.",
        config={
            "model": "gpt-4o-mini",
            "temperature": 0.1
        },
        labels=["extraction", "structured-data"]
    )
    print(f"Created prompt: {profile_prompt.name} (version {profile_prompt.version})")
    
    # 2. Personality Analysis Prompt  
    personality_prompt = langfuse.create_prompt(
        name="personality-analysis",
        prompt="Analyze the personality of this user based on their profile information. Determine their personality type (use MBTI or similar framework), key traits, communication style, and primary motivation. Also provide a confidence score (0.0 to 1.0) for your analysis.",
        config={
            "model": "gpt-4o-mini", 
            "temperature": 0.3
        },
        labels=["analysis", "psychology"]
    )
    print(f"Created prompt: {personality_prompt.name} (version {personality_prompt.version})")
    
    # 3. Recommendation Generation Prompt
    recommendation_prompt = langfuse.create_prompt(
        name="personalized-recommendations",
        prompt="""Generate personalized recommendations for activities, career advice, and lifestyle suggestions based on the user profile and personality analysis.

Provide specific, actionable recommendations in these categories:
- Activities: Hobbies and recreational activities that match their interests
- Career Advice: Professional development suggestions aligned with their goals
- Lifestyle Tips: Personal growth and wellness recommendations

Also provide a relevance score (0.0 to 1.0) indicating how well these recommendations match the user's profile.""",
        config={
            "model": "gpt-4o-mini",
            "temperature": 0.7
        },
        labels=["recommendations", "personalization"]
    )
    print(f"Created prompt: {recommendation_prompt.name} (version {recommendation_prompt.version})")
    
    print("\n✅ All prompts created successfully!")
    print("You can now manage and version these prompts in the Langfuse UI.")

if __name__ == "__main__":
    create_prompts()