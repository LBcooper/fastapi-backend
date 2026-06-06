"""POST /api/v1/chat — stub, wire up your LLM here."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Chat"])


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[list[ChatMessage]] = None
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    tokens_used: Optional[int]
    processing_time_ms: Optional[float]
    timestamp: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Stub — replace body with your LLM call."""
    return ChatResponse(
        reply="Chat endpoint active. Wire up your LLM here.",
        model="stub",
        tokens_used=None,
        processing_time_ms=None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
