"""
Streamlit app for Gemini API search interface with configurable prompt variables.
"""
import streamlit as st
import json
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import pickle
import os

from gemini_api import gemini_create_with_search_async


@dataclass
class SearchRequest:
    id: str
    prompt_variables: Dict[str, str]
    timestamp: datetime
    status: str  # 'pending', 'running', 'completed', 'error'
    result: Optional[str] = None
    error: Optional[str] = None


class SearchManager:
    def __init__(self):
        self.requests_file = "search_requests.pkl"
        self._lock = threading.Lock()
        
        # Initialize session state
        if 'search_requests' not in st.session_state:
            st.session_state.search_requests = self._load_requests()
        if 'active_searches' not in st.session_state:
            st.session_state.active_searches = {}

    def _load_requests(self) -> List[SearchRequest]:
        """Load requests from file if it exists."""
        if os.path.exists(self.requests_file):
            try:
                with open(self.requests_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return []
        return []

    def _save_requests(self, requests: List[SearchRequest]):
        """Save requests to file."""
        try:
            with open(self.requests_file, 'wb') as f:
                pickle.dump(requests, f)
        except:
            pass

    def add_request(self, prompt_variables: Dict[str, str]) -> str:
        request_id = str(uuid.uuid4())
        request = SearchRequest(
            id=request_id,
            prompt_variables=prompt_variables.copy(),
            timestamp=datetime.now(),
            status='pending'
        )
        
        with self._lock:
            st.session_state.search_requests.append(request)
            self._save_requests(st.session_state.search_requests)
        
        return request_id

    def get_request(self, request_id: str) -> Optional[SearchRequest]:
        with self._lock:
            requests = self._load_requests()  # Always load fresh from file
            for request in requests:
                if request.id == request_id:
                    return request
        return None

    def update_request_status(self, request_id: str, status: str, result: str = None, error: str = None):
        with self._lock:
            requests = self._load_requests()
            for request in requests:
                if request.id == request_id:
                    request.status = status
                    if result:
                        request.result = result
                    if error:
                        request.error = error
                    break
            
            self._save_requests(requests)
            # Also update session state for UI
            st.session_state.search_requests = requests

    async def run_search(self, request_id: str, model: str, prompt_name: str, prompt_variables: Dict[str, str]):
        try:
            self.update_request_status(request_id, 'running')
            
            result = await gemini_create_with_search_async(
                model=model,
                prompt_name=prompt_name,
                prompt_variables=prompt_variables
            )
            
            self.update_request_status(request_id, 'completed', result=result)
            
        except Exception as e:
            self.update_request_status(request_id, 'error', error=str(e))

    def start_search(self, request_id: str, model: str, prompt_name: str, prompt_variables: Dict[str, str]):
        def run_async_search():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_search(request_id, model, prompt_name, prompt_variables))
            finally:
                loop.close()

        thread = threading.Thread(target=run_async_search)
        thread.daemon = True
        thread.start()
        
        # Store thread reference in session state
        if 'active_searches' in st.session_state:
            st.session_state.active_searches[request_id] = thread


def load_config() -> Dict[str, Any]:
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Config file not found. Please ensure config.json exists.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Invalid JSON in config file.")
        st.stop()


def extract_json_from_response(response_text: str) -> tuple[Optional[Dict], str]:
    """Extract JSON from response text, handling prefix/suffix."""
    start = response_text.find('{')
    end = response_text.rfind('}') + 1
    
    if start != -1 and end > start:
        json_part = response_text[start:end]
        try:
            json_data = json.loads(json_part)
            return json_data, json_part
        except json.JSONDecodeError:
            return None, json_part
    return None, response_text


def json_to_markdown(json_data: Dict, level: int = 1) -> str:
    """Convert JSON to markdown with heading levels for indentation."""
    markdown = ""
    
    for key, value in json_data.items():
        heading = "#" * (level + 2)
        markdown += f"\n{heading} {key}\n\n"
        
        if isinstance(value, dict):
            markdown += json_to_markdown(value, level + 1)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    markdown += f"\n{heading}# Item {i + 1}\n\n"
                    markdown += json_to_markdown(item, level + 2)
                else:
                    markdown += f"- {item}\n"
            markdown += "\n"
        else:
            markdown += f"{value}\n\n"
    
    return markdown


def main():
    st.set_page_config(
        page_title="Gemini Search Interface",
        page_icon="🔍",
        layout="wide"
    )

    config = load_config()
    search_manager = SearchManager()

    st.title("🔍 Gemini Search Interface")

    # Create tabs
    input_tab, results_tab = st.tabs(["Search Input", "Results"])

    with input_tab:
        st.header("Search Parameters")
        
        # Create form for prompt variables
        with st.form("search_form"):
            prompt_variables = {}
            
            for var_config in config['prompt_variables']:
                label = var_config['label']
                name = var_config['name']
                placeholder = var_config.get('placeholder', f"Enter {label.lower()}")
                
                prompt_variables[name] = st.text_input(
                    label,
                    placeholder=placeholder,
                    key=f"input_{name}"
                )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                model = st.text_input(
                    "Model",
                    value=config.get('default_model', 'gemini-2.5-flash'),
                    key="model_input"
                )
                prompt_name = st.text_input(
                    "Prompt Name",
                    value=config.get('default_prompt_name', 'pombo-test-json-search'),
                    key="prompt_name_input"
                )
            
            submitted = st.form_submit_button("Start Search")
            
            if submitted:
                # Validate that at least one field is filled
                if not any(prompt_variables.values()):
                    st.error("Please fill in at least one field.")
                else:
                    # Filter out empty values
                    filtered_variables = {k: v for k, v in prompt_variables.items() if v.strip()}
                    
                    request_id = search_manager.add_request(filtered_variables)
                    search_manager.start_search(request_id, model, prompt_name, filtered_variables)
                    
                    st.success(f"Search started! Request ID: {request_id[:8]}...")
                    st.info("Check the Results tab to see progress.")

    with results_tab:
        st.header("Search Results")
        
        # Auto-refresh every 2 seconds
        if st.session_state.search_requests:
            time.sleep(0.1)  # Small delay to prevent too frequent updates
        
        # Refresh requests from file
        search_manager = SearchManager()
        st.session_state.search_requests = search_manager._load_requests()
        
        # Display all requests
        requests = sorted(st.session_state.search_requests, key=lambda x: x.timestamp, reverse=True)
        
        if not requests:
            st.info("No searches yet. Go to the Search Input tab to start a search.")
        else:
            for request in requests:
                with st.container():
                    # Status indicator
                    status_colors = {
                        'pending': '🔄',
                        'running': '⏳',
                        'completed': '✅',
                        'error': '❌'
                    }
                    
                    status_icon = status_colors.get(request.status, '❓')
                    
                    st.subheader(f"{status_icon} Request {request.id[:8]} - {request.status.title()}")
                    st.caption(f"Started: {request.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Show prompt variables
                    st.markdown("## Parameters:")
                    for key, value in request.prompt_variables.items():
                        st.markdown(f"- {key}: {value}")
                    
                    if request.status == 'completed' and request.result:
                        # Extract JSON and display results
                        json_data, raw_json = extract_json_from_response(request.result)
                        
                        # Raw response toggle
                        with st.expander("Raw Response"):
                            st.text(request.result)
                        
                        # JSON display
                        if json_data:
                            st.markdown("## Structured Response:")
                            markdown_content = json_to_markdown(json_data)
                            st.markdown(markdown_content)
                        else:
                            st.warning("Could not parse JSON from response")
                            with st.expander("Raw JSON Extract"):
                                st.text(raw_json)
                    
                    elif request.status == 'error':
                        st.error(f"Error: {request.error}")
                    
                    elif request.status in ['pending', 'running']:
                        st.info("Search in progress...")
                    
                    st.divider()
        
        # Auto-refresh button
        if st.button("🔄 Refresh Results"):
            st.rerun()


if __name__ == "__main__":
    main()