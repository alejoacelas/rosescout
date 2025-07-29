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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from rosescout.api.gpt import OpenAIClient, MCPTool


@dataclass
class SearchRequest:
    id: str
    prompt_variables: Dict[str, str]
    timestamp: datetime
    status: str  # 'pending', 'running', 'streaming', 'completed', 'error'
    web_search: bool = False
    result: Optional[str] = None
    error: Optional[str] = None
    custom_prompt: Optional[str] = None
    partial_result: Optional[str] = None  # For streaming responses


class SearchManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests = []

    def add_request(self, prompt_variables: Dict[str, str], web_search: bool = False, custom_prompt: Optional[str] = None) -> str:
        # Use first prompt variable value for request ID, or fallback to uuid
        first_value = next(iter(prompt_variables.values()), "") if prompt_variables else ""
        request_id = first_value[:15] + str(uuid.uuid4())[:6]
        request = SearchRequest(
            id=request_id,
            prompt_variables=prompt_variables.copy(),
            timestamp=datetime.now(),
            status='pending',
            web_search=web_search,
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

    def update_request_status(self, request_id: str, status: str, result: str = None, error: str = None, partial_result: str = None):
        with self._lock:
            for request in self._requests:
                if request.id == request_id:
                    request.status = status
                    if result:
                        request.result = result
                    if error:
                        request.error = error
                    if partial_result is not None:
                        request.partial_result = partial_result
                    break




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
        page_title="AI Background Check Assistant",
        page_icon="🔍",
        layout="wide"
    )

    config = load_config()
    search_manager = get_search_manager()

    st.title("🔍 Automated Background Check")

    # Sidebar for tool selection and advanced settings
    with st.sidebar:
        st.header("🔧 Tools & Settings")
        
        # Start new query button
        if st.button("🆕 Start New Query", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
        
        st.divider()
        
        # Tool group selection
        st.subheader("Available Tools")
        
        # Tool group selection
        tool_groups = config.get('mcp_tools_available', [])
        selected_tool_groups = []
        
        if tool_groups:
            for tool_group in tool_groups:
                if st.checkbox(
                    tool_group['label'],
                    value=True,  # Default to enabled
                    help=tool_group.get('description', ''),
                    key=f"tool_group_{tool_group['label']}"
                ):
                    selected_tool_groups.append(tool_group)
        
        # Web search option
        web_search_enabled = st.checkbox(
            "Web Search",
            value=config.get('default_web_search_enabled', True),
            help="Search public sources for customer information",
            key="web_search_enabled"
        )
        
        # Advanced settings
        st.subheader("Advanced Settings")
        
        # Model selection dropdown
        model_options = config.get('model_options', ['gpt-4.1'])
        default_model = config.get('default_model', 'gpt-4.1')
        default_index = model_options.index(default_model) if default_model in model_options else 0
        
        model = st.selectbox(
            "Model",
            options=model_options,
            index=default_index,
            key="model_select"
        )
        
        # System prompt (optional - if empty, uses default_prompt_id)
        system_prompt = st.text_area(
            "System Prompt (Optional)",
            value="",
            placeholder="Override the default instructions",
            key="system_prompt_input",
            help="If empty, the default prompt ID from config will be used. If provided, this custom prompt will override the default."
        )

    # Initialize session state for chat messages and conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None
    if "input_type" not in st.session_state:
        st.session_state.input_type = "initial"

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input handling - text area for initial input, chat_input for follow-ups
    if len(st.session_state.messages) == 0:
        # Initial input - use text area
        customer_info = st.text_area(
            "Paste customer information here...",
            height=150,
            placeholder="Enter customer details for background check analysis",
            key="initial_input"
        )
        
        if st.button("Analyze", type="primary"):
            st.session_state.pending_input = customer_info
            st.session_state.input_type = "initial"
            st.rerun()
    else:
        # Follow-up input - use chat_input
        if follow_up := st.chat_input("Ask additional questions..."):
            st.session_state.pending_input = follow_up
            st.session_state.input_type = "follow_up"
            st.rerun()

    # Process pending input
    if st.session_state.pending_input:
        user_input = st.session_state.pending_input
        input_type = st.session_state.input_type
        
        # Clear pending input
        st.session_state.pending_input = None
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message if it's a follow-up
        if input_type == "follow_up":
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # Build allowed tools list from selected tool groups
        allowed_tools = []
        for tool_group in selected_tool_groups:
            allowed_tools.extend(tool_group.get('tools', []))
        
        # Add web search to recommended tools if enabled
        if web_search_enabled:
            allowed_tools.append("web_search")
        
        # Prepend RECOMMENDED TOOLS to user input if tools are selected
        processed_user_input = user_input
        if allowed_tools:
            tools_list = ", ".join(allowed_tools)
            processed_user_input = f"RECOMMENDED TOOLS: [{tools_list}]\n{user_input}"
        
        # Create MCP tools from config
        mcp_tools = []
        mcp_servers = config.get('mcp_servers', [])
        for server in mcp_servers:
            if server.get('enabled_by_default', True):
                mcp_tools.append(MCPTool(
                    server_label=server['label'],
                    server_url=server['url'],
                    require_approval=server.get('require_approval', 'never')
                ))
        
        # Process the request
        if input_type == "follow_up":
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
        else:
            message_placeholder = st.empty()
            
        request_id = search_manager.add_request(
            prompt_variables={"Customer Information" if input_type == "initial" else "Follow-up Question": processed_user_input}, 
            web_search=web_search_enabled
        )
        
        # Run streaming search
        try:
            # Show initial thinking message
            thinking_msg = "💭 Processing (this may take 30-60 seconds)" if input_type == "initial" else "💭 Processing..."
            message_placeholder.markdown(thinking_msg)
            
            # Run the async function
            full_response = ""
            client = OpenAIClient()
            
            # Stream the response
            async def stream_response():
                nonlocal full_response
                search_manager.update_request_status(request_id, 'running')
                
                # Determine if we should use prompt_id or system_prompt
                prompt_id = None if system_prompt.strip() else config.get('default_prompt_id')
                actual_system_prompt = system_prompt.strip() if system_prompt.strip() else None
                
                async for delta in client.stream_content(
                    model=model,
                    system_prompt=actual_system_prompt,
                    prompt_id=prompt_id,
                    user_prompt=processed_user_input,
                    mcp_tools=mcp_tools if mcp_tools else None,
                    web_search=web_search_enabled,
                    previous_response_id=st.session_state.conversation_id
                ):
                    full_response += delta
                    # Update the placeholder with current response
                    message_placeholder.markdown(full_response + "▌")
                    # Update partial result in request
                    search_manager.update_request_status(request_id, 'streaming', partial_result=full_response)
                
                # Get complete response with annotations and tool calls
                complete_response = client.get_last_streaming_response()
                
                # Store response ID for conversation continuity
                if complete_response and complete_response.response_id:
                    st.session_state.conversation_id = complete_response.response_id
                
                # Format final response with annotations and tool calls
                final_response = full_response
                
                if complete_response:
                    # Add annotations as hyperlinks
                    if complete_response.annotations:
                        final_response += "\n\n**Sources:**\n"
                        # Track unique sources
                        seen_sources = set()
                        counter = 1
                        for annotation in complete_response.annotations:
                            if annotation.source:
                                if annotation.source not in seen_sources:
                                    final_response += f"{counter}. [{annotation.content}]({annotation.source})\n"
                                    seen_sources.add(annotation.source)
                                    counter += 1
                            else:
                                final_response += f"{counter}. {annotation.content}\n"
                                counter += 1
                    
                    # Add tool calls information
                    if complete_response.tool_calls:
                        final_response += "\n\n**Tools Used:**\n"
                        # Track unique tool names
                        seen_tools = set()
                        for tool_call in complete_response.tool_calls:
                            if tool_call.name not in seen_tools:
                                final_response += f"- {tool_call.name}\n"
                                seen_tools.add(tool_call.name)
                
                # Final update without cursor
                message_placeholder.markdown(final_response)
                search_manager.update_request_status(request_id, 'completed', result=final_response)
                
                return final_response
            
            # Run the async function
            result = asyncio.run(stream_response())
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })
            
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            message_placeholder.error(error_message)
            search_manager.update_request_status(request_id, 'error', error=str(e))
            
            # Add error to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
                    
        # Rerun to refresh the chat display
        st.rerun()


if __name__ == "__main__":
    main()