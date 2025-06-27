# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
streamlit run app.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Testing Tools
```bash
# Test Google Maps tools and Gemini integration
python test_tools.py
```

### Environment Setup
- Set `GEMINI_API_KEY` environment variable for Gemini API access
- Set `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` for Langfuse observability
- Set `GOOGLE_MAPS_API_KEY` for Google Maps API access

## Architecture

This is a Streamlit-based web application that provides a search interface for Gemini API with the following key components:

### Core Components
- **app.py**: Main Streamlit application with tabbed interface (Search Input, Results)
- **gemini_api.py**: Async Gemini API wrapper with search capabilities and Langfuse observability
- **json_utils.py**: JSON processing utilities for handling API responses
- **config.json**: Configuration file defining prompt variables and defaults

### Key Features
- **Async Search Management**: Uses threading to handle multiple concurrent search requests
- **JSON Response Processing**: Extracts, cleans, and displays JSON responses with reference handling
- **Langfuse Integration**: Full observability with prompt management and trace logging
- **Configurable Interface**: Dynamic form generation based on config.json settings

### Data Flow
1. User inputs are collected via dynamic Streamlit forms based on config.json
2. SearchManager handles request queuing and async execution
3. GeminiClient makes API calls with search tools (GoogleSearch, UrlContext)
4. Responses are processed through json_utils for display optimization
5. Results are shown in tabbed interface with original JSON, cleaned data, and references

### Configuration
- `config.json` defines prompt variables with labels and placeholders
- Default model and prompt name are configurable
- Prompt templates are managed through Langfuse

### Dependencies
- Streamlit for web interface
- google-genai for Gemini API integration
- Langfuse for observability and prompt management
- googlemaps for Google Maps API integration
- instructor, openai, python-dotenv for additional functionality

### Custom Tools System
- **tools.py**: Modular system for integrating external APIs with Gemini
- **GoogleMapsTools**: Provides geocoding and distance calculation functions
- **Custom tool integration**: Functions can be easily added to GeminiClient via `custom_tools` parameter
- **Independent testing**: Tools can be tested separately from Gemini integration using test_tools.py