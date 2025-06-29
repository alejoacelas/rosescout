"""
Streamlit app for Gemini API search interface with configurable prompt variables.
"""
import asyncio
import json
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st

from rosescout.api import GeminiClient
from rosescout.tools import get_coordinates, calculate_distance, web_search, screening_list_search, get_researcher_profile
from rosescout.utils import (
    extract_json_from_response, 
    limit_json_nesting_to_level2,
    extract_and_clean_json
)


@dataclass
class SearchRequest:
    id: str
    prompt_variables: Dict[str, str]
    timestamp: datetime
    status: str  # 'pending', 'running', 'completed', 'error'
    selected_tools: List[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    custom_prompt: Optional[str] = None
    
    def __post_init__(self):
        if self.selected_tools is None:
            self.selected_tools = []


class SearchManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests = []

    def add_request(self, prompt_variables: Dict[str, str], selected_tools: List[str] = None, custom_prompt: Optional[str] = None) -> str:
        # Use first prompt variable value for request ID, or fallback to uuid
        first_value = next(iter(prompt_variables.values()), "") if prompt_variables else ""
        request_id = first_value[:15] + str(uuid.uuid4())[:6]
        request = SearchRequest(
            id=request_id,
            prompt_variables=prompt_variables.copy(),
            timestamp=datetime.now(),
            status='pending',
            selected_tools=selected_tools or [],
            custom_prompt=custom_prompt
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

    async def run_search(self, request_id: str, model: str, prompt_name: str, prompt_variables: Dict[str, str], selected_tools: List[str], custom_prompt: Optional[str] = None):
        try:
            self.update_request_status(request_id, 'running')
            
            # Map tool names to actual functions
            tool_mapping = {
                'get_coordinates': get_coordinates,
                'calculate_distance': calculate_distance,
                'web_search': web_search,
                'screening_list_search': screening_list_search,
                'get_researcher_profile': get_researcher_profile
            }
            
            # Build tools list
            tools = []
            for tool_name in selected_tools:
                if tool_name in tool_mapping:
                    tools.append(tool_mapping[tool_name])
            
            client = GeminiClient()
            if custom_prompt:
                result = await client.generate_content(
                    model=model,
                    prompt=custom_prompt,
                    prompt_variables=prompt_variables,
                    tools=tools
                )
            else:
                result = await client.generate_content(
                    model=model,
                    prompt_name=prompt_name,
                    prompt_variables=prompt_variables,
                    tools=tools
                )
            
            self.update_request_status(request_id, 'completed', result=result)
            
        except Exception as e:
            self.update_request_status(request_id, 'error', error=str(e))

    def start_search(self, request_id: str, model: str, prompt_name: str, prompt_variables: Dict[str, str], selected_tools: List[str], custom_prompt: Optional[str] = None):
        def run_async_search():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_search(request_id, model, prompt_name, prompt_variables, selected_tools, custom_prompt))
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
        with open('config/config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Config file not found. Please ensure config/config.json exists.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Invalid JSON in config file.")
        st.stop()


def main():
    # Configure logging for Streamlit
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
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
            
            for i, var_config in enumerate(config['prompt_variables']):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Allow user to modify the key
                    custom_key = st.text_input(
                        "Variable Key",
                        value=var_config['name'],
                        key=f"key_{i}",
                        help="Edit the variable name/key"
                    )
                
                with col2:
                    label = var_config['label']
                    placeholder = var_config.get('placeholder', f"Enter {label.lower()}")
                    
                    prompt_variables[custom_key] = st.text_area(
                        label,
                        placeholder=placeholder,
                        key=f"input_{i}"
                    )
            
            # Tools selection
            st.subheader("🔧 Available Tools")
            st.caption("Select which tools Gemini can use for this search:")
            
            selected_tools = []
            available_tools = config.get('available_tools', [])
            
            # Create checkboxes for each available tool
            cols = st.columns(2)
            for i, tool in enumerate(available_tools):
                col = cols[i % 2]
                with col:
                    is_checked = tool.get('enabled_by_default', True)
                    if st.checkbox(
                        f"{tool['label']}",
                        value=is_checked,
                        help=tool.get('description', ''),
                        key=f"tool_{tool['name']}"
                    ):
                        selected_tools.append(tool['name'])
            
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
                custom_prompt = st.text_area(
                    "Custom Prompt",
                    placeholder="Enter a custom prompt to use instead of the Langfuse prompt. Leave empty to use the prompt name above.",
                    key="custom_prompt_input",
                    help="If filled, this will be used as the prompt instead of fetching from Langfuse."
                )
            
            submitted = st.form_submit_button("Start Search")
            
            if submitted:
                # Validate that at least one field is filled
                if not any(prompt_variables.values()):
                    st.warning("Prompt submited without any variables.")
                
                # Filter out empty values
                filtered_variables = {k: v for k, v in prompt_variables.items() if v.strip()}
                
                if not selected_tools:
                    st.warning("No tools selected. The search will run without any tools.")
                
                request_id = search_manager.add_request(filtered_variables, selected_tools=selected_tools, custom_prompt=custom_prompt if custom_prompt.strip() else None)
                search_manager.start_search(request_id, model, prompt_name, filtered_variables, selected_tools, custom_prompt if custom_prompt.strip() else None)
                
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
                    
                    st.subheader(f"{status_icon} {request.id[:15]} - {request.status.title()}")
                    st.caption(f"Started: {request.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Show prompt variables
                    parameters_toggle = st.expander("Input Parameters:", expanded=True)
                    for key, value in request.prompt_variables.items():
                        parameters_toggle.markdown(f"**{key.upper()}**")
                        parameters_toggle.markdown(f"{value}".replace("\n", "\n\n"))
                    
                    # Show selected tools
                    if request.selected_tools:
                        available_tools = config.get('available_tools', [])
                        tool_labels = [tool['label'] for tool in available_tools if tool['name'] in request.selected_tools]
                        parameters_toggle.markdown(f"**SELECTED TOOLS:** {', '.join(tool_labels)}")
                    else:
                        parameters_toggle.markdown("**SELECTED TOOLS:** None")
                    
                    if request.status == 'completed' and request.result:
                        # Extract JSON and display results
                        json_data, raw_json = extract_json_from_response(request.result)
                        
                        # JSON display
                        if json_data:
                            st.markdown("### Response:")
                            
                            # Extract references and clean JSON for separate display
                            cleaned_json, reference_fields = extract_and_clean_json(json_data)
                            
                            # Process cleaned JSON to limit nesting to level 2 (for Response tab)
                            level2_json_cleaned = limit_json_nesting_to_level2(cleaned_json)
                            
                            # Show original JSON in expander
                            st.markdown("**Original JSON**")
                            st.json(json_data, expanded=False)
                            
                            # Create tabs for different views
                            data_tab, references_tab = st.tabs([
                                "Response", 
                                "References", 
                            ])
                            
                            with data_tab:
                                try:
                                    st.dataframe(level2_json_cleaned, row_height=100)
                                except Exception as e:
                                    st.error(f"Cannot display as dataframe: {e}")
                            
                            with references_tab:
                                if reference_fields:
                                    st.dataframe(
                                        reference_fields,
                                        column_config={
                                            "path": "Field Path",
                                            "source": "Source",
                                            "url": st.column_config.LinkColumn(
                                                "URL",
                                                display_text="🔗 Open Link",
                                                width="small"
                                            )
                                        },
                                        hide_index=True,
                                        row_height=100
                                    )
                                else:
                                    st.info("No 'reference' fields found in the response.")
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