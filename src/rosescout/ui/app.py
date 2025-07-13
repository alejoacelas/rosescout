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
    mcp_servers: List[MCPTool] = None
    web_search: bool = False
    result: Optional[str] = None
    error: Optional[str] = None
    custom_prompt: Optional[str] = None
    partial_result: Optional[str] = None  # For streaming responses
    
    def __post_init__(self):
        if self.mcp_servers is None:
            self.mcp_servers = []


class SearchManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests = []

    def add_request(self, prompt_variables: Dict[str, str], mcp_servers: List[MCPTool] = None, web_search: bool = False, custom_prompt: Optional[str] = None) -> str:
        # Use first prompt variable value for request ID, or fallback to uuid
        first_value = next(iter(prompt_variables.values()), "") if prompt_variables else ""
        request_id = first_value[:15] + str(uuid.uuid4())[:6]
        request = SearchRequest(
            id=request_id,
            prompt_variables=prompt_variables.copy(),
            timestamp=datetime.now(),
            status='pending',
            mcp_servers=mcp_servers or [],
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

    async def run_search_streaming(self, request_id: str, model: str, system_prompt: str, user_prompt: str, mcp_servers: List[MCPTool], web_search: bool):
        """Run streaming search using OpenAI client."""
        try:
            self.update_request_status(request_id, 'running')
            
            client = OpenAIClient()
            
            # Stream the response
            full_response = ""
            async for delta in client.stream_content(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mcp_tools=mcp_servers,
                web_search=web_search
            ):
                full_response += delta
                # Update partial result in request
                self.update_request_status(request_id, 'streaming', partial_result=full_response)
            
            # Get complete response with annotations and tool calls
            complete_response = client.get_last_streaming_response()
            
            # Format final response with annotations and tool calls
            final_response = full_response
            
            if complete_response:
                # Add annotations as hyperlinks
                if complete_response.annotations:
                    final_response += "\n\n**Sources:**\n"
                    for i, annotation in enumerate(complete_response.annotations, 1):
                        if annotation.source:
                            final_response += f"{i}. [{annotation.content}]({annotation.source})\n"
                        else:
                            final_response += f"{i}. {annotation.content}\n"
                
                # Add tool calls information
                if complete_response.tool_calls:
                    final_response += "\n\n**Tools Used:**\n"
                    for tool_call in complete_response.tool_calls:
                        final_response += f"- {tool_call.name}\n"
            
            self.update_request_status(request_id, 'completed', result=final_response)
            
        except Exception as e:
            self.update_request_status(request_id, 'error', error=str(e))



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


def load_mcp_servers_from_config(config: Dict[str, Any]) -> List[MCPTool]:
    """Load MCP server configurations from config."""
    mcp_servers = []
    
    # Check for MCP servers in config
    mcp_config = config.get('mcp_servers', [])
    for server_config in mcp_config:
        mcp_servers.append(MCPTool(
            server_label=server_config['label'],
            server_url=server_config['url'],
            require_approval=server_config.get('require_approval', 'never')
        ))
    
    return mcp_servers


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
        page_title="RoseScout Chat",
        page_icon="🌹",
        layout="wide"
    )

    config = load_config()
    search_manager = get_search_manager()

    st.title("🌹 RoseScout Chat")

    # Sidebar for tool selection and advanced settings
    with st.sidebar:
        st.header("🔧 Tools & Settings")
        
        # Start new query button
        if st.button("🆕 Start New Query", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
        
        st.divider()
        
        # MCP servers and web search selection
        st.subheader("Available Tools")
        
        # Load MCP servers from config
        available_mcp_servers = load_mcp_servers_from_config(config)
        selected_mcp_servers = []
        
        # MCP servers checkboxes
        if available_mcp_servers:
            st.write("**MCP Servers:**")
            for server_config in config.get('mcp_servers', []):
                server = next((s for s in available_mcp_servers if s.server_label == server_config['label']), None)
                if server:
                    if st.checkbox(
                        f"{server.server_label}",
                        value=server_config.get('enabled_by_default', True),
                        help=server_config.get('description', f"Server URL: {server.server_url}"),
                        key=f"mcp_{server.server_label}"
                    ):
                        selected_mcp_servers.append(server)
        
        # Web search option
        web_search_enabled = st.checkbox(
            "Web Search",
            value=config.get('default_web_search_enabled', True),
            help="Enable web search functionality",
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

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if customer_info := st.chat_input("Enter customer information..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": customer_info})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(customer_info)
        
        # Add placeholder for assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Process the request
            request_id = search_manager.add_request(
                prompt_variables={"Customer Information": customer_info}, 
                mcp_servers=selected_mcp_servers,
                web_search=web_search_enabled
            )
            
            # Run streaming search directly (no threading)
            try:
                # Show initial thinking message
                message_placeholder.markdown("🤔 Processing...")
                
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
                        user_prompt=customer_info,
                        mcp_tools=selected_mcp_servers,
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
                            for i, annotation in enumerate(complete_response.annotations, 1):
                                if annotation.source:
                                    final_response += f"{i}. [{annotation.content}]({annotation.source})\n"
                                else:
                                    final_response += f"{i}. {annotation.content}\n"
                        
                        # Add tool calls information
                        if complete_response.tool_calls:
                            final_response += "\n\n**Tools Used:**\n"
                            for tool_call in complete_response.tool_calls:
                                final_response += f"- {tool_call.name}\n"
                    
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