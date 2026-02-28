"""
Dialogue Session Manager — Multi-turn flow persistence and resumption.

When a dialogue flow hits a `collect` step (waiting for user input),
the session manager saves the flow's state so it can be resumed
when the inbound response arrives.

This enables multi-turn agentic conversations:
  1. Flow sends a question → hits collect step → session saved
  2. User replies → orchestrator detects active session → resumes flow
  3. User response is evaluated (intent/entity extraction) → next step runs
  4. Flow continues from the branch point with the user's answer in context

Sessions also track:
  - Which flow is running
  - The execution context at the point of pause
  - Timeout tracking (how long to wait before the flow auto-advances)
  - Conversation-scoped flow memory (accumulated across turns)
"""
from __future__ import annotations

import structlog
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from pydantic import BaseModel, Field

from models.schemas import ChannelType

logger = structlog.get_logger()


# ──────────────────────────────────────────────────────
#  Session Model
# ──────────────────────────────────────────────────────

class DialogueSession(BaseModel):
    """Persisted state of a paused dialogue flow."""
    id: str                                         # session_id = conversation_id
    flow_id: str
    paused_at_step: str                             # step_id where we paused (collect step)
    channel: ChannelType
    execution_context: dict[str, Any] = {}          # full context at time of pause
    message_buffer: list[str] = []                  # accumulated message fragments
    expected_intents: list[str] = []                # what we're waiting for
    entity_extraction: list[str] = []               # entities to pull from response
    timeout_at: Optional[datetime] = None           # when to auto-advance
    timeout_goto: str = ""                          # step to jump to on timeout
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resumed_count: int = 0                          # how many times resumed (multi-collect flows)

    @property
    def is_expired(self) -> bool:
        if not self.timeout_at:
            return False
        return datetime.now(timezone.utc) >= self.timeout_at


# ──────────────────────────────────────────────────────
#  Session Store
# ──────────────────────────────────────────────────────

class DialogueSessionStore:
    """
    In-memory session store. Keyed by conversation_id.
    Only one active session per conversation at a time.
    """

    def __init__(self):
        self._sessions: dict[str, DialogueSession] = {}

    def save(self, session: DialogueSession):
        self._sessions[session.id] = session
        logger.info("dialogue_session_saved",
                     session_id=session.id,
                     flow_id=session.flow_id,
                     paused_at=session.paused_at_step)

    def get(self, conversation_id: str) -> Optional[DialogueSession]:
        session = self._sessions.get(conversation_id)
        if session and session.is_expired:
            logger.info("dialogue_session_expired",
                         session_id=session.id,
                         flow_id=session.flow_id)
            # Don't remove — the executor will handle timeout_goto
            return session
        return session

    def remove(self, conversation_id: str):
        removed = self._sessions.pop(conversation_id, None)
        if removed:
            logger.info("dialogue_session_removed",
                         session_id=removed.id,
                         flow_id=removed.flow_id)

    def has_active_session(self, conversation_id: str) -> bool:
        session = self._sessions.get(conversation_id)
        return session is not None and not session.is_expired

    def list_active(self) -> list[DialogueSession]:
        now = datetime.now(timezone.utc)
        return [s for s in self._sessions.values()
                if not s.timeout_at or s.timeout_at > now]

    def list_expired(self) -> list[DialogueSession]:
        now = datetime.now(timezone.utc)
        return [s for s in self._sessions.values()
                if s.timeout_at and s.timeout_at <= now]

    @property
    def count(self) -> int:
        return len(self._sessions)


# ──────────────────────────────────────────────────────
#  Intent Matcher — classify user response against expected intents
# ──────────────────────────────────────────────────────

# Keyword-based intent classification (fast, no LLM needed)
INTENT_KEYWORDS: dict[str, list[str]] = {
    "confirmed": ["yes", "confirm", "ok", "sure", "agreed", "accept", "done", "paid", "haan", "ha"],
    "denied": ["no", "reject", "cancel", "refuse", "decline", "nahi", "nope"],
    "payment_promise": ["will pay", "promise", "by tomorrow", "by monday", "next week", "sending now", "transferring"],
    "payment_issue": ["can't pay", "unable", "difficulty", "financial", "extension", "delay"],
    "complaint": ["unhappy", "complaint", "bad", "terrible", "worst", "disappointed", "angry", "issue"],
    "positive_feedback": ["good", "great", "excellent", "happy", "satisfied", "amazing", "awesome", "love"],
    "question": ["what", "when", "how", "where", "why", "which", "can you", "could you"],
    "escalate": ["manager", "supervisor", "escalate", "senior", "higher", "speak to"],
    "reschedule": ["reschedule", "later", "postpone", "another time", "not now"],
}


def classify_intent(
    message: str,
    expected_intents: list[str] = None,
) -> dict[str, Any]:
    """
    Classify user message intent using keyword matching.

    Returns:
        dict with keys: matched_intent, confidence, all_scores
    """
    message_lower = message.lower().strip()
    scores: dict[str, float] = {}

    search_intents = expected_intents if expected_intents else list(INTENT_KEYWORDS.keys())

    for intent in search_intents:
        keywords = INTENT_KEYWORDS.get(intent, [])
        if not keywords:
            continue
        matches = sum(1 for kw in keywords if kw in message_lower)
        scores[intent] = matches / len(keywords) if keywords else 0

    if not scores or max(scores.values()) == 0:
        return {
            "matched_intent": "unknown",
            "confidence": 0.0,
            "all_scores": scores,
        }

    best = max(scores, key=scores.get)
    return {
        "matched_intent": best,
        "confidence": scores[best],
        "all_scores": scores,
    }


def extract_entities(
    message: str,
    entity_types: list[str],
) -> dict[str, Any]:
    """
    Extract simple entities from message text.

    Supported types:
      - amount: numbers that look like money
      - date: date-like strings
      - reference_number: alphanumeric codes
      - phone: phone number patterns
    """
    import re
    entities: dict[str, Any] = {}

    if "amount" in entity_types:
        # Match currency amounts: ₹5000, $1,234.56, 5000, Rs. 5000
        amounts = re.findall(r'[₹$]?\s*[\d,]+\.?\d*', message)
        if amounts:
            # Clean and parse the first match
            cleaned = amounts[0].replace('₹', '').replace('$', '').replace(',', '').strip()
            try:
                entities["amount"] = float(cleaned)
            except ValueError:
                pass

    if "date" in entity_types:
        # Simple date patterns
        dates = re.findall(
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4}',
            message, re.IGNORECASE)
        if dates:
            entities["date"] = dates[0]

    if "reference_number" in entity_types:
        refs = re.findall(r'[A-Z]{2,4}[-#]?\d{4,}', message.upper())
        if refs:
            entities["reference_number"] = refs[0]

    if "phone" in entity_types:
        phones = re.findall(r'\+?\d[\d\s-]{8,}\d', message)
        if phones:
            entities["phone"] = phones[0].strip()

    return entities


async def llm_classify_intent(
    message: str,
    expected_intents: list[str],
    context: dict[str, Any] = None,
    llm_generate: Any = None,
) -> dict[str, Any]:
    """
    Use LLM for intelligent intent classification when:
    - Keyword matching returns low confidence
    - The message is ambiguous or complex
    - Expected intents are domain-specific

    Falls back to keyword classification if LLM is unavailable.
    """
    if not llm_generate:
        return classify_intent(message, expected_intents)

    context = context or {}
    intents_str = ", ".join(expected_intents) if expected_intents else "any relevant intent"

    try:
        import json as _json
        response = await llm_generate(
            "You are an intent classifier for a business conversation agent. "
            "Analyze the user's message and classify their intent. "
            "Respond with ONLY valid JSON.",
            f"Expected intents: {intents_str}\n"
            f"Business context: {_json.dumps(context.get('business_context', {}))}\n"
            f"User message: {message}\n\n"
            f"Return JSON: {{\"matched_intent\": \"...\", \"confidence\": 0.0-1.0, "
            f"\"reasoning\": \"brief explanation\"}}",
            {"max_tokens": 200, "temperature": 0.2},
        )
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        result = _json.loads(text)
        result["all_scores"] = {result.get("matched_intent", "unknown"): result.get("confidence", 0.0)}
        return result
    except Exception:
        return classify_intent(message, expected_intents)


async def llm_extract_entities(
    message: str,
    entity_types: list[str],
    context: dict[str, Any] = None,
    llm_generate: Any = None,
) -> dict[str, Any]:
    """
    Use LLM for intelligent entity extraction.

    Handles complex cases regex can't:
    - "I'll pay by next Tuesday" → date extraction with relative parsing
    - "About two and a half thousand" → amount = 2500
    - "My order INV slash twenty-four slash oh-three" → reference_number
    - Multilingual content and informal expressions

    Falls back to regex extraction if LLM is unavailable.
    """
    if not llm_generate:
        return extract_entities(message, entity_types)

    context = context or {}

    try:
        import json as _json
        types_desc = ", ".join(entity_types)
        response = await llm_generate(
            "You are an entity extraction system. Extract structured data "
            "from the user's message. Be precise with amounts and dates. "
            "Respond with ONLY valid JSON.",
            f"Entity types to extract: {types_desc}\n"
            f"Business context: {_json.dumps(context.get('business_context', {}))}\n"
            f"User message: {message}\n\n"
            f"Return JSON with extracted entities. Use null for entities not found. "
            f"For amounts, return as numbers. For dates, return as YYYY-MM-DD.",
            {"max_tokens": 200, "temperature": 0.1},
        )
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        result = _json.loads(text)
        # Filter out nulls
        return {k: v for k, v in result.items() if v is not None}
    except Exception:
        return extract_entities(message, entity_types)
