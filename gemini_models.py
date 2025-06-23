"""
Pydantic models for Gemini API responses.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class GeminiMessage(BaseModel):
    content: str
    role: str


class GeminiChoice(BaseModel):
    message: GeminiMessage
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class GeminiUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GroundingSource(BaseModel):
    title: Optional[str] = None
    uri: Optional[str] = None


class GroundingMetadata(BaseModel):
    queries: Optional[List[str]] = None
    sources: List[GroundingSource] = []


class GeminiMetadata(BaseModel):
    grounding: GroundingMetadata


class GeminiResponse(BaseModel):
    id: Optional[str] = None
    object: str
    created: Optional[int] = None
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[GeminiChoice]
    usage: GeminiUsage
    gemini_metadata: GeminiMetadata