"""
Instructor client setup that enables structured outputs from Gemini with search.
This wraps the Gemini search functionality to work with Instructor for Pydantic models.
"""
import os
from dotenv import load_dotenv
import instructor
from google import genai
from instructor import Instructor
from gemini_wrapper import gemini_create_with_search

# Load environment variables
load_dotenv()

def load_instructor_client(model: str = "gemini-2.5-pro", use_search: bool = False) -> instructor.Instructor:
    """
    Initialize and return an Instructor client configured for the specified model.
    
    Args:
        model: The Gemini model name to use
        use_search: Whether to enable Google Search integration
        
    Returns:
        Configured Instructor client
        
    Raises:
        ValueError: If API key is not set or use_search is True for non-Gemini models
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    if use_search:
        # Use our custom Gemini patching implementation with search
        client = Instructor(
            client=None,
            create=instructor.patch(
                create=gemini_create_with_search,
                mode=instructor.Mode.JSON,
            )
        )
    else:
        # Use standard Gemini client without search
        client = instructor.from_genai(genai.Client(api_key=api_key))

    return client