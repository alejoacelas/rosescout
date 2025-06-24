"""
Streamlit app for Gemini API search interface with configurable prompt variables.
"""
import asyncio
import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st

from gemini_api import GeminiClient
from json_utils import (
    extract_json_from_response, 
    limit_json_nesting_to_level2,
    extract_reference_fields,
    extract_url_fields
)


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
        self._lock = threading.Lock()
        self._requests = []

    def add_request(self, prompt_variables: Dict[str, str]) -> str:
        request_id = prompt_variables.get('person', "")[:10] + str(uuid.uuid4())[:6]
        request = SearchRequest(
            id=request_id,
            prompt_variables=prompt_variables.copy(),
            timestamp=datetime.now(),
            status='pending'
        )
        
        with self._lock:
            self._requests.append(request)
        
        return request_id

    def get_request(self, request_id: str) -> Optional[SearchRequest]:
        with self._lock:
            for request in self._requests:
                if request.id == request_id:
                    return request
        return None

    def get_all_requests(self) -> List[SearchRequest]:
        """Get all requests sorted by timestamp."""
        with self._lock:
            return sorted(self._requests, key=lambda x: x.timestamp, reverse=True)

    def update_request_status(self, request_id: str, status: str, result: str = None, error: str = None):
        with self._lock:
            for request in self._requests:
                if request.id == request_id:
                    request.status = status
                    if result:
                        request.result = result
                    if error:
                        request.error = error
                    break

    async def run_search(self, request_id: str, model: str, prompt_name: str, prompt_variables: Dict[str, str]):
        try:
            self.update_request_status(request_id, 'running')
            
            client = GeminiClient()
            result = await client.generate_with_search(
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


@st.cache_resource
def get_search_manager() -> SearchManager:
    """Get a singleton SearchManager instance."""
    return SearchManager()


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





def main():
    st.set_page_config(
        page_title="Gemini Search Interface",
        page_icon="🔍",
        layout="wide"
    )

    config = load_config()
    search_manager = get_search_manager()

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
                    
                    st.success(f"Search started! Request ID: {request_id}...")
                    st.info("Check the Results tab to see progress.")

    with results_tab:
        st.header("Search Results")
        
        if st.button("🔄 Refresh Results", use_container_width=True):
            st.rerun()
        
        # Get all requests from the singleton search manager
        requests = search_manager.get_all_requests()
        
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
                            
                            # Process JSON to limit nesting to level 2
                            level2_json = limit_json_nesting_to_level2(json_data)
                            
                            st.markdown("### JSON Structure (limited to 2 levels)")
                            st.json(level2_json)
                            
                            # Create tabs for different views
                            data_tab, references_tab, urls_tab = st.tabs([
                                "📊 Data Table", 
                                "📚 References", 
                                "🔗 URLs"
                            ])
                            
                            with data_tab:
                                st.markdown("### Full Data Table")
                                try:
                                    st.dataframe(level2_json, row_height=100)
                                except Exception as e:
                                    st.error(f"Cannot display as dataframe: {e}")
                            
                            with references_tab:
                                st.markdown("### Reference Fields")
                                reference_fields = extract_reference_fields(json_data)
                                if reference_fields:
                                    st.dataframe(
                                        reference_fields,
                                        column_config={
                                            "path": "Field Path",
                                            "value": "Reference Value"
                                        },
                                        hide_index=True,
                                        row_height=100
                                    )
                                else:
                                    st.info("No 'reference' fields found in the response.")
                            
                            with urls_tab:
                                st.markdown("### URL Fields")
                                url_fields = extract_url_fields(json_data)
                                if url_fields:
                                    st.dataframe(
                                        url_fields,
                                        column_config={
                                            "path": "Field Path",
                                            "value": st.column_config.LinkColumn(
                                                "URL",
                                                help="Click to open link",
                                                display_text="🔗 Open Link"
                                            )
                                        },
                                        hide_index=True,
                                        row_height=60
                                    )
                                else:
                                    st.info("No 'url' fields found in the response.")
                        else:
                            st.warning("Could not parse JSON from response")
                            with st.expander("Raw JSON Extract"):
                                st.text(raw_json)
                    
                    elif request.status == 'error':
                        st.error(f"Error: {request.error}")
                    
                    elif request.status in ['pending', 'running']:
                        st.info("Search in progress...")
                    
                    st.divider()


if __name__ == "__main__":
    main()