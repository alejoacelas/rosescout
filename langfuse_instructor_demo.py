"""Langfuse + Instructor demo showcasing structured data extraction and AI analysis."""
from typing import List

from dotenv import load_dotenv
from langfuse import observe, get_client
from langfuse.openai import OpenAI
import instructor
from pydantic import BaseModel, Field

load_dotenv()

# Use Langfuse OpenAI integration and patch with instructor
client = instructor.patch(OpenAI())

# Initialize Langfuse client for scoring
langfuse = get_client()

class UserProfile(BaseModel):
    """User profile data model."""
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    interests: List[str] = Field(description="List of hobbies and interests")
    location: str = Field(description="Current city or location")
    occupation: str = Field(description="Job title or profession")

class PersonalityAnalysis(BaseModel):
    """Personality analysis data model."""
    personality_type: str = Field(description="Primary personality type (e.g., MBTI)")
    key_traits: List[str] = Field(description="List of dominant personality traits")
    communication_style: str = Field(description="How this person typically communicates")
    motivation: str = Field(description="What drives and motivates this person")
    confidence_score: float = Field(
        description="Confidence in the analysis (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

@observe(as_type="generation")
def extract_user_profile(text: str) -> UserProfile:
    """Extract structured user profile from unstructured text."""
    # Get prompt from Langfuse (specify version since it's not in production yet)
    prompt = langfuse.get_prompt("user-profile-extraction", version=1)
    
    # Update current generation with prompt info
    langfuse.update_current_generation(prompt=prompt)
    
    return client.chat.completions.create(
        model=prompt.config.get("model", "gpt-4o-mini"),
        temperature=prompt.config.get("temperature", 0.1),
        response_model=UserProfile,
        messages=[
            {"role": "system", "content": prompt.prompt},
            {"role": "user", "content": text}
        ],
        langfuse_prompt=prompt
    )

@observe(as_type="generation")
def analyze_personality(profile: UserProfile) -> PersonalityAnalysis:
    """Analyze personality based on user profile."""
    # Get prompt from Langfuse (specify version since it's not in production yet)
    prompt = langfuse.get_prompt("personality-analysis", version=1)
    
    # Update current generation with prompt info
    langfuse.update_current_generation(prompt=prompt)
    
    return client.chat.completions.create(
        model=prompt.config.get("model", "gpt-4o-mini"),
        temperature=prompt.config.get("temperature", 0.3),
        response_model=PersonalityAnalysis,
        messages=[
            {"role": "system", "content": prompt.prompt},
            {"role": "user", "content": f"Profile: {profile.model_dump_json()}"}
        ],
        langfuse_prompt=prompt
    )

class Recommendations(BaseModel):
    """Structured recommendations model."""
    activities: List[str] = Field(description="Recommended activities and hobbies")
    career_advice: List[str] = Field(description="Professional development suggestions")
    lifestyle_tips: List[str] = Field(description="Lifestyle and personal growth tips")
    relevance_score: float = Field(
        description="How relevant these recommendations are (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

@observe(as_type="generation")
def generate_personalized_recommendation(
    profile: UserProfile, analysis: PersonalityAnalysis
) -> Recommendations:
    """Generate personalized recommendations based on profile and personality analysis."""
    # Get prompt from Langfuse (specify version since it's not in production yet)
    prompt = langfuse.get_prompt("personalized-recommendations", version=1)
    
    # Update current generation with prompt info
    langfuse.update_current_generation(prompt=prompt)
    
    return client.chat.completions.create(
        model=prompt.config.get("model", "gpt-4o-mini"),
        temperature=prompt.config.get("temperature", 0.7),
        response_model=Recommendations,
        messages=[
            {"role": "system", "content": prompt.prompt},
            {
                "role": "user",
                "content": (
                    f"User Profile: {profile.model_dump_json()}\n"
                    f"Personality Analysis: {analysis.model_dump_json()}"
                )
            }
        ],
        langfuse_prompt=prompt
    )

def main():
    """Main function that orchestrates the entire workflow."""

    sample_text = """
    Hi, I'm Sarah, a 28-year-old software engineer living in San Francisco.
    I love hiking, reading sci-fi novels, and experimenting with new cooking recipes.
    I work at a tech startup and I'm really passionate about artificial intelligence and machine learning.
    In my free time, I also enjoy playing board games with friends and learning new programming languages.
    """

    print("🚀 Starting Langfuse + Instructor Demo")
    print("=" * 50)

    # Step 1: Extract structured profile
    print("📊 Extracting user profile...")
    profile = extract_user_profile(sample_text)
    print(f"Profile: {profile.model_dump_json(indent=2)}")
    print()

    # Step 2: Analyze personality
    print("🧠 Analyzing personality...")
    analysis = analyze_personality(profile)
    print(f"Analysis: {analysis.model_dump_json(indent=2)}")
    print()

    # Step 3: Generate recommendations
    print("💡 Generating personalized recommendations...")
    recommendations = generate_personalized_recommendation(profile, analysis)
    print(f"\n🎯 Activities: {recommendations.activities}")
    print(f"💼 Career Advice: {recommendations.career_advice}")
    print(f"🌱 Lifestyle Tips: {recommendations.lifestyle_tips}")
    print(f"📊 Relevance Score: {recommendations.relevance_score}")

    # Score the results in Langfuse
    trace_id = langfuse.get_current_trace_id()
    
    # Score personality analysis confidence
    langfuse.create_score(
        trace_id=trace_id,
        name="personality-confidence",
        value=analysis.confidence_score,
        comment="Confidence in personality analysis"
    )
    
    # Score recommendation relevance
    langfuse.create_score(
        trace_id=trace_id,
        name="recommendation-relevance",
        value=recommendations.relevance_score,
        comment="Relevance of generated recommendations"
    )
    
    # Flush to ensure scores are sent
    langfuse.flush()

    print("\n✅ Demo completed! Check your Langfuse dashboard for tracing details.")

    return {
        "profile": profile,
        "analysis": analysis,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    result = main()
