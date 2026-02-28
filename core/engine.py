"""
Conversation Engine — LLM-powered response generation.

Takes conversation context and generates appropriate responses
using Claude. Handles:
- Context-aware response generation
- Intent detection (resolved, needs escalation, etc.)
- Conversation summarization
- Multi-turn dialogue management
"""
from __future__ import annotations

import json
import structlog
from typing import Any, Optional

from config.settings import get_settings
from models.schemas import ChannelType

logger = structlog.get_logger()


class ConversationEngine:
    """
    Generates contextual responses using Claude or OpenAI.
    Adapts tone and length based on the channel being used.
    Supports both Anthropic and OpenAI LLM providers.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = None
        self._provider = getattr(self._settings.llm, "provider", "anthropic")

    @property
    def is_openai(self) -> bool:
        return self._provider == "openai"

    async def _get_client(self):
        if self._client is None:
            try:
                if self.is_openai:
                    from openai import AsyncOpenAI
                    self._client = AsyncOpenAI(
                        api_key=self._settings.llm.api_key
                    )
                    logger.info("llm_client_initialized", provider="openai",
                                model=self._settings.llm.model)
                else:
                    import anthropic
                    self._client = anthropic.AsyncAnthropic(
                        api_key=self._settings.llm.api_key
                    )
                    logger.info("llm_client_initialized", provider="anthropic",
                                model=self._settings.llm.model)
            except Exception as e:
                logger.error("llm_client_init_failed", provider=self._provider, error=str(e))
                self._client = None
        return self._client

    async def _call_llm(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """Unified LLM call that handles both Anthropic and OpenAI APIs."""
        client = await self._get_client()
        if not client:
            return ""

        max_tokens = max_tokens or self._settings.llm.max_tokens
        temperature = temperature if temperature is not None else self._settings.llm.temperature

        if self.is_openai:
            # OpenAI: system prompt is a message in the messages list
            oai_messages = [{"role": "system", "content": system}] + messages
            response = await client.chat.completions.create(
                model=self._settings.llm.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=oai_messages,
            )
            return response.choices[0].message.content
        else:
            # Anthropic: system prompt is a separate parameter
            response = await client.messages.create(
                model=self._settings.llm.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            return response.content[0].text

    async def generate_response(
        self,
        context: dict[str, Any],
        channel: ChannelType,
        user_message: str = "",
    ) -> str:
        """
        Generate a contextual response for the conversation.

        Args:
            context: Full conversation context from ContextTracker
            channel: Current channel (affects tone/length)
            user_message: The latest message from the contact (if any)
        """
        system_prompt = self._build_system_prompt(context, channel)
        messages = self._build_messages(context, user_message)

        try:
            result = await self._call_llm(
                system=system_prompt,
                messages=messages,
            )
            if not result:
                return self._fallback_response(context, channel)
            return result
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            return self._fallback_response(context, channel)

    async def detect_intent(self, context: dict[str, Any], user_message: str) -> dict[str, Any]:
        """
        Analyze the contact's message to determine intent.
        Returns structured intent data.
        """
        system = """Analyze the user's message in the context of a business follow-up conversation.
Return a JSON object with:
- intent: one of the intents listed below
- confidence: float 0-1
- extracted_info: dict of any useful information extracted (dates, amounts, reasons, etc.)
- sentiment: positive | neutral | negative
- suggested_action: what the agent should do next

INTENT VOCABULARY (use EXACTLY one of these):
  Payment: payment_confirmed | payment_promised | will_pay | paying_soon | payment_delayed
  Acknowledgement: acknowledged | understood | will_check
  Order: order_accepted | cancel_order | cancelled
  Issues: issue | problem | damaged | wrong_item | complaint | bad_experience | unhappy
  Resolution: resolved | satisfied | all_good | issue_fixed
  Feedback: feedback_given | rating_provided | review
  Escalation: escalation_needed | speak_to_manager | dispute
  Callback: request_callback
  General: positive_response | negative_response | question | unclear

Pick the MOST SPECIFIC intent that matches. For example, if someone says "I already paid yesterday",
use "payment_confirmed" not "positive_response".

Return ONLY valid JSON, no other text."""

        try:
            result = await self._call_llm(
                system=system,
                messages=[
                    {"role": "user", "content": (
                        f"Business context: {json.dumps(context.get('business_context', {}))}\n\n"
                        f"Active business processes: {json.dumps(context.get('state_bindings', []))}\n\n"
                        f"Conversation history: {json.dumps(context.get('conversation_history', []))}\n\n"
                        f"Latest message from contact: {user_message}"
                    )},
                ],
                max_tokens=500,
                temperature=0.3,
            )
            if not result:
                return {"intent": "unknown", "confidence": 0.0}
            result = result.strip()
            # Parse JSON response
            if result.startswith("```"):
                result = result.split("```")[1].strip()
                if result.startswith("json"):
                    result = result[4:].strip()
            return json.loads(result)
        except Exception as e:
            logger.error("intent_detection_failed", error=str(e))
            return {"intent": "unknown", "confidence": 0.0, "sentiment": "neutral"}

    async def summarize_conversation(self, context: dict[str, Any]) -> str:
        """Generate a concise summary of the conversation."""
        try:
            result = await self._call_llm(
                system="Summarize this business conversation in 2-3 sentences. Focus on: what was discussed, any commitments made, and the current status.",
                messages=[
                    {"role": "user", "content": f"Business context: {json.dumps(context.get('business_context', {}))}\n\nConversation: {json.dumps(context.get('conversation_history', []))}"},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return result or "Conversation summary unavailable."
        except Exception as e:
            logger.error("summarization_failed", error=str(e))
            return "Unable to generate summary."

    # ── Step-Level Generation (for Dialogue Executor) ──

    async def generate_for_step(
        self,
        system_prompt: str,
        user_prompt: str,
        constraints: dict[str, Any] = None,
    ) -> str:
        """
        Generate text for a single dialogue flow step.
        Called by DialogueExecutor for 'generate' steps.

        Args:
            system_prompt: Step-specific system prompt (already interpolated)
            user_prompt:   Step-specific user prompt (already interpolated)
            constraints:   Optional overrides (max_tokens, temperature, etc.)
        """
        constraints = constraints or {}
        try:
            result = await self._call_llm(
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=constraints.get("max_tokens"),
                temperature=constraints.get("temperature"),
            )
            return result or user_prompt
        except Exception as e:
            logger.error("step_generation_failed", error=str(e))
            return user_prompt

    # ── Prompt Construction ───────────────────────────────────

    def _build_system_prompt(self, context: dict[str, Any], channel: ChannelType) -> str:
        """Build the system prompt with business context, state info, and channel adaptations."""
        contact = context.get("contact", {})
        business = context.get("business_context", {})
        followup = context.get("followup", {})
        state_bindings = context.get("state_bindings", [])

        # Channel-specific instructions
        channel_instructions = {
            ChannelType.CHAT: "Keep responses conversational and moderate length. Use markdown sparingly.",
            ChannelType.EMAIL: "Write in professional email style. Can be slightly longer and more detailed.",
            ChannelType.WHATSAPP: "Keep messages SHORT (2-4 sentences max). Use simple language. Can use emojis sparingly.",
            ChannelType.VOICE: "Responses will be spoken via TTS. Keep them very short (1-3 sentences). Use natural spoken language.",
        }

        # Build state context block
        state_context = ""
        if state_bindings:
            state_lines = []
            for sb in state_bindings:
                state_lines.append(
                    f"  - {sb['process_type']}: currently in '{sb['current_state']}' state"
                    f" (entity: {sb.get('entity_type','?')} {sb.get('entity_id','')})"
                    f"{' [RESOLVED]' if sb.get('is_terminal') else ''}"
                )
                if sb.get("last_transition"):
                    state_lines.append(f"    Last transition: {sb['last_transition']}")
            state_context = "\n\nBusiness Process States:\n" + "\n".join(state_lines)

        template = self._settings.llm.system_prompt_template or self._default_system_prompt()

        return template.replace(
            "{{business_domain}}", self._settings.business_domain
        ).replace(
            "{{contact_name}}", contact.get("name", "the contact")
        ).replace(
            "{{contact_role}}", contact.get("role", "")
        ).replace(
            "{{business_context}}", json.dumps(business, indent=2)
        ).replace(
            "{{conversation_history}}", json.dumps(context.get("conversation_history", []), indent=2)
        ).replace(
            "{{followup_reason}}", followup.get("reason", "General follow-up")
        ).replace(
            "{{state_context}}", state_context
        ) + f"\n\nCHANNEL: {channel.value}\n{channel_instructions.get(channel, '')}"

    def _build_messages(self, context: dict[str, Any], user_message: str) -> list[dict[str, str]]:
        """Build the message history for the LLM."""
        messages = []
        for entry in context.get("conversation_history", [])[-10:]:
            role = "assistant" if entry["role"] == "agent" else "user"
            messages.append({"role": role, "content": entry["content"]})

        if user_message:
            messages.append({"role": "user", "content": user_message})

        # Ensure messages alternate and start with user
        if not messages:
            messages = [{"role": "user", "content": "Please initiate the follow-up conversation."}]
        elif messages[0]["role"] == "assistant":
            messages.insert(0, {"role": "user", "content": "[Conversation started]"})

        return messages

    def _default_system_prompt(self) -> str:
        return """You are a professional follow-up agent for {{business_domain}}.
You are contacting {{contact_name}} ({{contact_role}}) regarding business matters.

Business Context:
{{business_context}}
{{state_context}}

Previous Conversation:
{{conversation_history}}

Follow-up Reason: {{followup_reason}}

GUIDELINES:
- Be professional, concise, and action-oriented
- Reference specific details from the business context
- Be aware of the current business process state and tailor your response accordingly
- If the contact has concerns, acknowledge them empathetically
- Always aim for a clear next step or resolution
- If the matter is resolved, confirm and thank them
- If you cannot help, offer to connect them with the right person"""

    def _fallback_response(self, context: dict[str, Any], channel: ChannelType) -> str:
        """Generate a basic template response when LLM is unavailable."""
        contact_name = context.get("contact", {}).get("name", "there")
        reason = context.get("followup", {}).get("reason", "your recent activity")

        if channel == ChannelType.WHATSAPP:
            return f"Hi {contact_name}! Following up regarding {reason}. Could you please provide an update? 🙏"
        elif channel == ChannelType.VOICE:
            return f"Hi {contact_name}, I'm calling to follow up regarding {reason}. Do you have a moment?"
        else:
            return f"Hi {contact_name},\n\nI'm reaching out to follow up regarding {reason}. Could you please let us know the current status?\n\nThank you for your time."
