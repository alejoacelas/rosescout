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
# Or install as package in development mode:
pip install -e .
```

### Testing Tools
```bash
# Test tools and Gemini integration
python tests/test_tools.py
```

### Environment Setup
- Set `GEMINI_API_KEY` environment variable for Gemini API access
- Set `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` for Langfuse observability
- Set `GOOGLE_MAPS_API_KEY` for Google Maps API access
- Set `TAVILY_SEARCH_API_KEY` for web search functionality
- Set `CONSOLIDATED_SCREENING_LIST_API_KEY` for screening list searches

## Architecture

This is a modular Streamlit-based web application that provides a search interface for Gemini API with the following structure:

### Directory Structure
```
rosescout/
├── src/rosescout/           # Main package
│   ├── api/                 # API integration modules
│   │   ├── __init__.py
│   │   └── gemini.py        # Gemini API client
│   ├── tools/               # External API tools
│   │   ├── __init__.py
│   │   ├── base.py          # Base classes
│   │   ├── maps.py          # Google Maps integration
│   │   ├── search.py        # Tavily search integration
│   │   └── screening.py     # Screening list integration
│   ├── utils/               # Data processing utilities
│   │   ├── __init__.py
│   │   └── json_utils.py    # JSON handling utilities
│   └── ui/                  # User interface
│       ├── __init__.py
│       └── app.py           # Streamlit application
├── tests/                   # Test files
│   └── test_tools.py
├── config/                  # Configuration files
│   └── config.json
├── docs/                    # Documentation
├── prompts/                 # Prompt templates
├── app.py                   # Main entry point
└── setup.py                 # Package setup
```

### Core Components
- **src/rosescout/ui/app.py**: Main Streamlit application with tabbed interface
- **src/rosescout/api/gemini.py**: Async Gemini API wrapper with Langfuse observability
- **src/rosescout/utils/json_utils.py**: JSON processing utilities
- **config/config.json**: Configuration file defining prompt variables and defaults

### Key Features
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Async Search Management**: Threading for concurrent search requests
- **JSON Response Processing**: Advanced JSON cleaning and reference extraction
- **Langfuse Integration**: Full observability with prompt management
- **Configurable Interface**: Dynamic form generation based on configuration

### Tool System
The modular tool system allows easy integration of external APIs:

- **src/rosescout/tools/maps.py**: Google Maps geocoding and distance calculation
- **src/rosescout/tools/search.py**: Tavily web search integration
- **src/rosescout/tools/screening.py**: Consolidated Screening List API
- **Base classes**: Common error handling and patterns in `base.py`

Tools can be easily added to GeminiClient via the `tools` parameter and tested independently.

### Data Flow
1. User inputs collected via dynamic Streamlit forms
2. SearchManager handles request queuing and async execution
3. GeminiClient makes API calls with selected tools
4. Responses processed through JSON utilities for display
5. Results shown in tabbed interface with data and references

### Configuration
- `config/config.json` defines prompt variables and tool settings
- Default model and prompt names configurable
- Prompt templates managed through Langfuse
- Environment variables for API keys and settings