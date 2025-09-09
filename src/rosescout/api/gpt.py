"""
OpenAI API wrapper with streaming and async support.
"""
import os
import logging
import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, AsyncGenerator

from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call from the AI response."""
    name: str
    arguments: Dict[str, Any]
    output: Optional[Any] = None


@dataclass
class Annotation:
    """Represents an annotation (e.g., from web search)."""
    type: str
    content: str
    source: Optional[str] = None


@dataclass
class MCPTool:
    """MCP tool configuration."""
    server_label: str
    server_url: str
    require_approval: str = "never"  # 'always' or 'never'

@dataclass
class AIResponse:
    """Structured response from OpenAI API."""
    text: str
    tool_calls: List[ToolCall]
    annotations: List[Annotation]
    response_id: Optional[str] = None  # For conversation continuity
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None


class OpenAIClient:
    """Simple OpenAI API client with streaming and async support."""
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.sync_client = OpenAI(api_key=self.api_key)
        self._last_streaming_response = None
    
    def _build_tools(
        self, mcp_tools: Optional[List[MCPTool]] = None, web_search: bool = False
    ) -> Optional[List[Dict]]:
        """Build tools array for OpenAI Responses API."""
        tools = []
        # Add MCP tools (passed as MCPTool dataclasses)
        if mcp_tools:
            for mcp_tool in mcp_tools:
                tools.append({
                    "type": "mcp",
                    "server_label": mcp_tool.server_label,
                    "server_url": mcp_tool.server_url,
                    "require_approval": mcp_tool.require_approval
                })
        # Add web search if requested
        if web_search:
            tools.append({
                "type": "web_search"
            })
        return tools if tools else None
    
    def _extract_response_data(self, response) -> AIResponse:
        """Extract all data from response into structured format.
        
        Handles both direct response objects and dictionary representations
        from streaming events.
        """
        tool_calls = []
        annotations = []
        text = ""
        
        # Handle both object and dict formats
        if isinstance(response, dict):
            output_data = response.get('output', [])
            usage_data = response.get('usage')
            model_data = response.get('model')
        else:
            output_data = getattr(response, 'output', [])
            usage_data = getattr(response, 'usage', None)
            model_data = getattr(response, 'model', None)
        
        # Extract from response.output array (correct API structure)
        if output_data:
            for output_item in output_data:
                # Handle both dict and object formats for output items
                item_type = output_item.get('type') if isinstance(output_item, dict) else getattr(output_item, 'type', None)
                
                # Handle message type - main text content
                if item_type == 'message':
                    # Extract text and annotations from message content
                    content = output_item.get('content', []) if isinstance(output_item, dict) else getattr(output_item, 'content', [])
                    for content_item in content:
                        content_type = content_item.get('type') if isinstance(content_item, dict) else getattr(content_item, 'type', None)
                        if content_type == 'output_text':
                            text = content_item.get('text', '') if isinstance(content_item, dict) else getattr(content_item, 'text', '')

                            # Extract annotations (e.g., URL citations)
                            annotations_data = content_item.get('annotations', []) if isinstance(content_item, dict) else getattr(content_item, 'annotations', [])
                            for annotation in annotations_data:
                                if isinstance(annotation, dict):
                                    annotations.append(Annotation(
                                        type=annotation.get('type', 'unknown'),
                                        content=annotation.get('title', '') or annotation.get('text', ''),
                                        source=annotation.get('url', '') or annotation.get('source', '')
                                    ))
                                else:
                                    annotations.append(Annotation(
                                        type=getattr(annotation, 'type', 'unknown'),
                                        content=(getattr(annotation, 'title', '') or
                                                 getattr(annotation, 'text', '')),
                                        source=(getattr(annotation, 'url', '') or
                                                getattr(annotation, 'source', ''))
                                    ))
                
                # Handle web search calls
                elif item_type == 'web_search_call':
                    if isinstance(output_item, dict):
                        tool_calls.append(ToolCall(
                            name='web_search',
                            arguments={
                                'query': output_item.get('query', ''),
                                'status': output_item.get('status', 'unknown')
                            },
                            output=output_item.get('results', [])
                        ))
                    else:
                        tool_calls.append(ToolCall(
                            name='web_search',
                            arguments={
                                'query': getattr(output_item, 'query', ''),
                                'status': getattr(output_item, 'status', 'unknown')
                            },
                            output=getattr(output_item, 'output', [])
                        ))
                
                # Handle MCP calls
                elif item_type == 'mcp_call':
                    if isinstance(output_item, dict):
                        tool_calls.append(ToolCall(
                            name=output_item.get('name', 'mcp_tool'),
                            arguments=output_item.get('arguments', {}),
                            output=output_item.get('output', None)
                        ))
                    else:
                        tool_calls.append(ToolCall(
                            name=getattr(output_item, 'name', 'mcp_tool'),
                            arguments=getattr(output_item, 'arguments', {}),
                            output=getattr(output_item, 'output', None)
                        ))
                
                # Handle function calls
                elif item_type == 'function_call':
                    if isinstance(output_item, dict):
                        tool_calls.append(ToolCall(
                            name=output_item.get('name', 'function'),
                            arguments=output_item.get('arguments', {}),
                            output=output_item.get('output', None)
                        ))
                    else:
                        tool_calls.append(ToolCall(
                            name=getattr(output_item, 'name', 'function'),
                            arguments=getattr(output_item, 'arguments', {}),
                            output=getattr(output_item, 'output', None)
                        ))
        if not text:
            text = getattr(response, 'text', '')
        # Extract response ID for conversation continuity
        response_id = None
        if isinstance(response, dict):
            response_id = response.get('id')
        else:
            response_id = getattr(response, 'id', None)
            
        return AIResponse(
            text=text,
            tool_calls=tool_calls,
            annotations=annotations,
            response_id=response_id,
            usage={
                "total_tokens": (usage_data.get('total_tokens') if isinstance(usage_data, dict) 
                               else getattr(usage_data, 'total_tokens', None) if usage_data else None),
                "input_tokens": (usage_data.get('input_tokens') if isinstance(usage_data, dict)
                                else getattr(usage_data, 'input_tokens', None) if usage_data else None),
                "output_tokens": (usage_data.get('output_tokens') if isinstance(usage_data, dict)
                                 else getattr(usage_data, 'output_tokens', None) if usage_data else None),
            } if usage_data else None,
            model=model_data
        )
    
    async def generate_content(
        self,
        *,
        model: str = "gpt-4.1",
        system_prompt: Optional[str] = None,
        prompt_id: Optional[str] = None,
        user_prompt: str,
        mcp_tools: Optional[List[MCPTool]] = None,
        web_search: bool = False,
        previous_response_id: Optional[str] = None
    ) -> AIResponse:
        """
        Generate content using OpenAI Responses API.
        
        Args:
            model: OpenAI model name
            system_prompt: System prompt - overrides prompt_id
            prompt_id: Prompt ID to use instead of system_prompt
            user_prompt: User prompt
            mcp_tools: List of MCPTool dataclasses
            web_search: Whether to use web search
            previous_response_id: Previous response ID for conversation continuity
            
        Returns:
            AIResponse with text, tool calls, and annotations
        """
        logger.info("🤖 OpenAI call - Model: %s", model)

        tools = self._build_tools(mcp_tools, web_search)

        if mcp_tools:
            tool_labels = [tool.server_label for tool in mcp_tools]
            logger.info("🔧 MCP tools: %s", ', '.join(tool_labels))
        if web_search:
            logger.info("🔍 Web search enabled")
        # Build request parameters
        request_params = {
            "model": model,
            "input": user_prompt,
            "reasoning": {"effort": "medium"},
        }

        # Use prompt ID if provided, otherwise use system_prompt
        if system_prompt:
            request_params["instructions"] = system_prompt
        elif prompt_id:
            request_params["prompt"] = {"id": prompt_id}

        if tools:
            request_params["tools"] = tools
            
        if previous_response_id:
            request_params["previous_response_id"] = previous_response_id
        # Make API call using responses API
        response = await self.client.responses.create(**request_params)

        # Extract all data from response
        ai_response = self._extract_response_data(response)

        # Log tool usage
        if ai_response.tool_calls:
            for tool_call in ai_response.tool_calls:
                logger.info("🔧 Tool called: %s", tool_call.name)
                logger.info("   Arguments: %s", tool_call.arguments)
        return ai_response
    
    async def stream_content(
        self,
        *,
        model: str = "gpt-4.1",
        system_prompt: Optional[str] = None,
        prompt_id: Optional[str] = None,
        user_prompt: str,
        mcp_tools: Optional[List[MCPTool]] = None,
        web_search: bool = False,
        previous_response_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream content using OpenAI Responses API.
        
        Args:
            model: OpenAI model name
            system_prompt: System prompt (becomes instructions) - optional if prompt_id is provided
            prompt_id: Prompt ID to use instead of system_prompt
            user_prompt: User prompt (becomes input)
            mcp_tools: List of MCPTool dataclasses
            web_search: Whether to use web search
            previous_response_id: Previous response ID for conversation continuity
            
        Yields:
            Text deltas as they arrive
        """
        logger.info("🤖 OpenAI streaming call - Model: %s", model)

        tools = self._build_tools(mcp_tools, web_search)

        if mcp_tools:
            tool_labels = [tool.server_label for tool in mcp_tools]
            logger.info("🔧 MCP tools: %s", ', '.join(tool_labels))
        if web_search:
            logger.info("🔍 Web search enabled")
        # Build request parameters
        request_params = {
            "model": model,
            "input": user_prompt,
            "stream": True,
            "reasoning": {"effort": "medium"},
        }

        # Use prompt ID if provided, otherwise use system_prompt
        if prompt_id:
            request_params["prompt"] = {"id": prompt_id}
        elif system_prompt:
            request_params["instructions"] = system_prompt

        if tools:
            request_params["tools"] = tools
            
        if previous_response_id:
            request_params["previous_response_id"] = previous_response_id
        # Make streaming API call using responses API
        stream = await self.client.responses.create(**request_params)
        
        complete_response = None
        
        async for event in stream:
            # Extract content from event based on responses API streaming format
            event_type = getattr(event, 'type', None)
            
            # Capture the complete response when streaming finishes
            if event_type == 'response.completed':
                # Extract the complete response data from the event
                if hasattr(event, 'response'):
                    complete_response = event.response
                elif hasattr(event, 'data'):
                    # Handle case where response is in JSON format
                    try:
                        event_data = json.loads(event.data) if isinstance(event.data, str) else event.data
                        complete_response = event_data.get('response')
                    except (json.JSONDecodeError, AttributeError):
                        pass
            
            # Yield text deltas for streaming
            elif event_type == 'response.output_text.delta':
                delta = getattr(event, 'delta', None)
                if delta:
                    yield delta
            elif event_type == 'response.output_text.done':
                # Final text chunk - usually not needed as deltas provide complete text
                pass
            elif hasattr(event, 'choices') and event.choices:
                # Handle Chat Completions streaming format as fallback
                delta = event.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content
        
        # Store the complete response for later retrieval
        self._last_streaming_response = complete_response

    def get_last_streaming_response(self) -> Optional[AIResponse]:
        """
        Get the complete response from the last stream_content call.
        
        This method returns the structured response data that was captured
        during the most recent call to stream_content(). You must call
        stream_content() first to populate the response data.
        
        Returns:
            AIResponse with text, tool calls, and annotations from the last
            streaming response, or None if no streaming response was captured.
        """
        if self._last_streaming_response:
            return self._extract_response_data(self._last_streaming_response)
        else:
            return None


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of how to use the OpenAI client."""
        client = OpenAIClient()
        
        # Example 1: Basic async request with web search
        print("=== Async Request with Web Search ===")
        response = await client.generate_content(
            model="gpt-4.1",
            system_prompt="You are a helpful assistant.",
            user_prompt="What are the latest AI developments in 2025?",
            web_search=True
        )
        
        print(f"Response: {response.text}")
        print(f"Usage: {response.usage}")
        print(f"Tool calls: {response.tool_calls}")
        print(f"Annotations: {len(response.annotations)}")
        for annotation in response.annotations:
            print(f"  - {annotation.type}: {annotation.content} ({annotation.source})")
        
        # Example 2: MCP server usage
        print("\n=== MCP Server Request ===")

        # Create MCP tool configurations
        mcp_tools = [
            MCPTool(
                server_label="cf-template",
                server_url="https://cf-template.alejoacelas.workers.dev/sse",
                require_approval="never"
            ),
        ]

        response = await client.generate_content(
            model="gpt-4.1",
            user_prompt="What's the distance between NY and LA? Use tools",
            mcp_tools=mcp_tools
        )

        print(f"Response: {response.text}")
        print(f"Tool calls: {len(response.tool_calls)}")
        # Example 3: Streaming request
        print("\n=== Streaming Request ===")
        print("Streaming response: ", end="", flush=True)

        async for delta in client.stream_content(
            model="gpt-4.1",
            system_prompt="You are a helpful assistant.",
            user_prompt="What's the distance between NY and LA? Use tools",
            mcp_tools=mcp_tools
        ):
            print(delta, end="", flush=True)

        print("\n\nStreaming complete!")
        
        # Example 4: Get complete response from last streaming call
        print("\n=== Get Last Streaming Response ===")
        complete_response = client.get_last_streaming_response()
        
        if complete_response:
            print(f"Complete response captured from stream:")
            print(f"Text length: {len(complete_response.text)} characters")
            print(f"Usage: {complete_response.usage}")
            print(f"Tool calls: {complete_response.tool_calls}")
            print(f"Annotations: {len(complete_response.annotations)}")
            if complete_response.annotations:
                print("Sample annotations:")
                for i, annotation in enumerate(complete_response.annotations[:3]):
                    print(f"  {i+1}. {annotation.type}: {annotation.source}")
        else:
            print("No streaming response captured.")

    # Run example
    asyncio.run(example_usage())