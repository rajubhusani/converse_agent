"""
Turn-Taking Engine — Natural conversation flow management.

Models the rhythm of a human phone conversation: who speaks when,
when to yield the floor, when to backchannel, and how to handle
interruptions gracefully.

Key behaviors:
- Adaptive endpointing: adjusts pause-to-finalize based on context
  (questions need shorter pauses, mid-sentence needs longer)
- Backchannel generation: inserts "mm-hmm", "I see", "right" at
  natural points so the contact knows the agent is listening
- Filler detection: recognizes "um", "uh", "like" from the contact
  and treats them as still-speaking signals (doesn't interrupt)
- Barge-in handling: when the contact interrupts, immediately stops
  TTS and yields the floor
- Turn prediction: uses cues like rising intonation, complete
  sentences, and pauses to predict when to speak next
- Silence management: escalating responses from gentle prompt
  to "are you there?" to graceful goodbye
"""
from __future__ import annotations

import time
import structlog
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
#  CONVERSATION STATE
# ══════════════════════════════════════════════════════════════

class FloorHolder(str, Enum):
    """Who currently 'has the floor' in the conversation."""
    AGENT = "agent"           # agent is speaking or about to speak
    CONTACT = "contact"       # contact is speaking
    YIELDED = "yielded"       # floor is open — next speaker takes it
    TRANSITIONING = "transitioning"  # brief gap between speakers


class ConversationPhase(str, Enum):
    """Where in the conversation arc we are."""
    GREETING = "greeting"
    OPENING = "opening"       # stating purpose
    BODY = "body"             # main discussion
    NEGOTIATING = "negotiating"  # back-and-forth on terms
    CLOSING = "closing"
    FAREWELL = "farewell"


class SilenceLevel(str, Enum):
    """Escalating silence response levels."""
    NONE = "none"
    GENTLE_PROMPT = "gentle_prompt"       # 5s: backchannel or soft prompt
    CHECK_IN = "check_in"                 # 10s: "Are you still there?"
    FINAL_CHECK = "final_check"           # 20s: "I think we may have lost connection"
    GOODBYE = "goodbye"                   # 30s: graceful end


# ══════════════════════════════════════════════════════════════
#  BACKCHANNEL CONFIGURATION
# ══════════════════════════════════════════════════════════════

@dataclass
class BackchannelConfig:
    """Configuration for automatic backchannel generation."""
    enabled: bool = True
    min_interval_ms: int = 4000          # don't backchannel more often than this
    trigger_after_words: int = 15         # backchannel after N words from contact
    trigger_after_pause_ms: int = 800     # brief pause → backchannel opportunity

    # Response pools by context
    acknowledgment: list[str] = field(default_factory=lambda: [
        "Mm-hmm.", "I see.", "Right.", "Okay.", "Got it.", "Understood.",
    ])
    encouragement: list[str] = field(default_factory=lambda: [
        "Go on.", "Tell me more.", "I'm listening.", "Sure.",
    ])
    empathy: list[str] = field(default_factory=lambda: [
        "I understand.", "That makes sense.", "Of course.",
        "I appreciate that.", "Absolutely.",
    ])

    def get_response(self, context: str = "acknowledgment", index: int = 0) -> str:
        pool = getattr(self, context, self.acknowledgment)
        return pool[index % len(pool)]


# ══════════════════════════════════════════════════════════════
#  ENDPOINTING CONFIGURATION
# ══════════════════════════════════════════════════════════════

@dataclass
class EndpointingConfig:
    """
    Adaptive endpointing: how long to wait after speech stops before
    considering the turn complete.

    Shorter for questions (the answer is expected quickly) and longer
    for mid-sentence pauses (the person is thinking).
    """
    base_ms: int = 300                 # default pause-to-finalize
    question_ms: int = 200             # after agent asks a question → shorter wait
    mid_sentence_ms: int = 600         # detected mid-thought → wait longer
    after_filler_ms: int = 800         # "um" / "uh" → they're thinking
    number_dictation_ms: int = 1200    # dictating numbers → long pauses normal
    emotional_ms: int = 500            # emotional content → be patient
    min_ms: int = 150                  # absolute minimum
    max_ms: int = 2000                 # absolute maximum

    def compute(self, context: dict[str, Any]) -> int:
        """Compute adaptive endpointing delay based on conversation context."""
        base = self.base_ms

        if context.get("agent_asked_question"):
            base = self.question_ms
        if context.get("filler_detected"):
            base = max(base, self.after_filler_ms)
        if context.get("mid_sentence"):
            base = max(base, self.mid_sentence_ms)
        if context.get("dictating_numbers"):
            base = max(base, self.number_dictation_ms)
        if context.get("emotional_content"):
            base = max(base, self.emotional_ms)

        return max(self.min_ms, min(base, self.max_ms))


# ══════════════════════════════════════════════════════════════
#  FILLER WORD DETECTION
# ══════════════════════════════════════════════════════════════

# Common filler words that indicate the speaker is still formulating
FILLER_WORDS = {
    "um", "uh", "uhh", "umm", "hmm", "hm", "er", "erm", "ah",
    "like", "you know", "I mean", "sort of", "kind of", "basically",
    "well", "so", "actually", "honestly", "let me think",
}

# Sentence-ending indicators (suggest turn completion)
SENTENCE_ENDERS = {".", "!", "?", "right?", "okay?", "yeah?", "no?"}


def detect_fillers(text: str) -> list[str]:
    """Detect filler words/phrases in transcript text using word boundaries."""
    import re
    text_lower = text.lower().strip()
    found = []
    for filler in FILLER_WORDS:
        # Use word boundary matching to avoid false positives
        # e.g. "ah" should not match "ahead"
        if " " in filler:
            # Multi-word fillers: simple substring is fine
            if filler in text_lower:
                found.append(filler)
        else:
            # Single-word fillers: require word boundaries
            pattern = r'\b' + re.escape(filler) + r'\b'
            if re.search(pattern, text_lower):
                found.append(filler)
    return found


def is_likely_complete(text: str) -> bool:
    """Heuristic: does this text look like a complete thought?"""
    text = text.strip()
    if not text:
        return False
    # Ends with sentence-ending punctuation
    if text[-1] in ".!?":
        return True
    # Very short responses (yes/no/ok)
    if len(text.split()) <= 3 and text.lower().rstrip(".,!?") in {
        "yes", "no", "okay", "ok", "sure", "fine", "right", "thanks",
        "thank you", "got it", "understood", "yep", "nope", "yeah", "nah",
    }:
        return True
    return False


def is_question(text: str) -> bool:
    """Detect if text is a question."""
    text = text.strip()
    if text.endswith("?"):
        return True
    lower = text.lower()
    question_starts = ("can ", "could ", "would ", "will ", "do ", "does ", "did ",
                       "is ", "are ", "was ", "were ", "have ", "has ", "what ",
                       "when ", "where ", "who ", "why ", "how ", "shall ")
    return any(lower.startswith(q) for q in question_starts)


# ══════════════════════════════════════════════════════════════
#  SILENCE ESCALATION
# ══════════════════════════════════════════════════════════════

@dataclass
class SilenceConfig:
    """Configurable silence timeout escalation."""
    gentle_prompt_s: float = 5.0
    check_in_s: float = 10.0
    final_check_s: float = 20.0
    goodbye_s: float = 30.0

    # Response templates
    gentle_prompts: list[str] = field(default_factory=lambda: [
        "Take your time.",
        "I'm here whenever you're ready.",
    ])
    check_in_prompts: list[str] = field(default_factory=lambda: [
        "Are you still there?",
        "Hello? Can you hear me?",
    ])
    final_check_prompts: list[str] = field(default_factory=lambda: [
        "I think we may have lost connection. I'll try calling back.",
        "It seems like the connection might have dropped.",
    ])
    goodbye_prompts: list[str] = field(default_factory=lambda: [
        "Alright, I'll follow up with a message. Have a good day!",
        "No worries, I'll reach out again later. Goodbye!",
    ])

    def get_level(self, silence_seconds: float) -> SilenceLevel:
        if silence_seconds >= self.goodbye_s:
            return SilenceLevel.GOODBYE
        if silence_seconds >= self.final_check_s:
            return SilenceLevel.FINAL_CHECK
        if silence_seconds >= self.check_in_s:
            return SilenceLevel.CHECK_IN
        if silence_seconds >= self.gentle_prompt_s:
            return SilenceLevel.GENTLE_PROMPT
        return SilenceLevel.NONE

    def get_prompt(self, level: SilenceLevel, index: int = 0) -> str:
        pools = {
            SilenceLevel.GENTLE_PROMPT: self.gentle_prompts,
            SilenceLevel.CHECK_IN: self.check_in_prompts,
            SilenceLevel.FINAL_CHECK: self.final_check_prompts,
            SilenceLevel.GOODBYE: self.goodbye_prompts,
        }
        pool = pools.get(level, [])
        return pool[index % len(pool)] if pool else ""


# ══════════════════════════════════════════════════════════════
#  INTERRUPTION HANDLER
# ══════════════════════════════════════════════════════════════

class InterruptionStrategy(str, Enum):
    IMMEDIATE_YIELD = "immediate_yield"     # stop TTS immediately
    FINISH_SENTENCE = "finish_sentence"     # complete current sentence, then yield
    IGNORE = "ignore"                       # continue speaking (rare)


@dataclass
class InterruptionConfig:
    """How to handle contact interrupting the agent."""
    default_strategy: InterruptionStrategy = InterruptionStrategy.IMMEDIATE_YIELD
    min_speech_before_yield_ms: int = 300    # ignore very brief noises
    acknowledge_interruption: bool = True     # say "sorry, go ahead" after yielding
    max_ignored_interruptions: int = 2        # after this many ignores, always yield

    acknowledgments: list[str] = field(default_factory=lambda: [
        "Sorry, go ahead.",
        "Yes, please go ahead.",
        "Of course, I'm listening.",
    ])


# ══════════════════════════════════════════════════════════════
#  TURN-TAKING STATE MACHINE
# ══════════════════════════════════════════════════════════════

class TurnTakingEngine:
    """
    Manages the conversational turn-taking state.

    Tracks who has the floor, when to backchannel, when to speak,
    and how to handle silences and interruptions naturally.
    """

    def __init__(
        self,
        backchannel: BackchannelConfig = None,
        endpointing: EndpointingConfig = None,
        silence: SilenceConfig = None,
        interruption: InterruptionConfig = None,
    ):
        self.backchannel = backchannel or BackchannelConfig()
        self.endpointing = endpointing or EndpointingConfig()
        self.silence = silence or SilenceConfig()
        self.interruption = interruption or InterruptionConfig()

        # State
        self._floor: FloorHolder = FloorHolder.AGENT  # agent speaks first in outbound
        self._phase: ConversationPhase = ConversationPhase.GREETING
        self._last_speaker_change: float = time.monotonic()
        self._last_backchannel: float = 0.0
        self._contact_word_count: int = 0
        self._silence_start: Optional[float] = None
        self._last_silence_level: SilenceLevel = SilenceLevel.NONE
        self._pending_barge_in: bool = False
        self._agent_asked_question: bool = False
        self._ignored_interruptions: int = 0
        self._filler_detected: bool = False

    # ── State access ──────────────────────────────────────────

    @property
    def floor(self) -> FloorHolder:
        return self._floor

    @property
    def phase(self) -> ConversationPhase:
        return self._phase

    def set_phase(self, phase: ConversationPhase) -> None:
        self._phase = phase
        logger.debug("conversation_phase_changed", phase=phase.value)

    # ── Events ────────────────────────────────────────────────

    def on_agent_speech_start(self) -> None:
        """Agent begins speaking (TTS started)."""
        self._floor = FloorHolder.AGENT
        self._silence_start = None
        self._last_silence_level = SilenceLevel.NONE

    def on_agent_speech_end(self, text: str = "") -> None:
        """Agent finishes speaking (TTS completed)."""
        self._floor = FloorHolder.YIELDED
        self._last_speaker_change = time.monotonic()
        self._agent_asked_question = is_question(text)
        self._silence_start = time.monotonic()

    def on_contact_speech_start(self) -> dict[str, Any]:
        """
        Contact begins speaking. Returns action dict.
        If agent was speaking, this is a barge-in.
        """
        result: dict[str, Any] = {"action": "listen"}
        self._silence_start = None
        self._last_silence_level = SilenceLevel.NONE
        self._filler_detected = False

        if self._floor == FloorHolder.AGENT:
            # Barge-in!
            strategy = self._resolve_interruption_strategy()
            if strategy == InterruptionStrategy.IMMEDIATE_YIELD:
                self._floor = FloorHolder.CONTACT
                result = {
                    "action": "barge_in",
                    "strategy": "immediate_yield",
                    "stop_tts": True,
                    "acknowledge": self.interruption.acknowledge_interruption,
                }
                self._ignored_interruptions = 0
            elif strategy == InterruptionStrategy.FINISH_SENTENCE:
                result = {"action": "barge_in", "strategy": "finish_sentence", "stop_tts": False}
            else:
                self._ignored_interruptions += 1
                result = {"action": "continue_speaking"}
        else:
            self._floor = FloorHolder.CONTACT

        self._contact_word_count = 0
        self._last_speaker_change = time.monotonic()
        return result

    def on_contact_speech_end(self, text: str = "") -> dict[str, Any]:
        """
        Contact stops speaking. Returns action dict with endpointing advice.
        """
        self._floor = FloorHolder.TRANSITIONING
        words = text.split() if text else []
        self._contact_word_count += len(words)
        self._filler_detected = bool(detect_fillers(text)) if text else False

        # Compute adaptive endpointing
        endpoint_context = {
            "agent_asked_question": self._agent_asked_question,
            "filler_detected": self._filler_detected,
            "mid_sentence": not is_likely_complete(text) if text else False,
            "dictating_numbers": any(c.isdigit() for c in (text or "")),
        }
        endpoint_ms = self.endpointing.compute(endpoint_context)

        result: dict[str, Any] = {
            "action": "wait",
            "endpointing_ms": endpoint_ms,
            "likely_complete": is_likely_complete(text) if text else False,
        }

        # Should we backchannel?
        if self._should_backchannel():
            result["backchannel"] = self._pick_backchannel()

        self._silence_start = time.monotonic()
        return result

    def on_silence_tick(self) -> Optional[dict[str, Any]]:
        """
        Called periodically during silence. Returns escalation action if needed.
        """
        if self._silence_start is None:
            return None
        if self._floor == FloorHolder.AGENT:
            return None  # Agent is about to speak, ignore silence

        elapsed = time.monotonic() - self._silence_start
        level = self.silence.get_level(elapsed)

        if level == self._last_silence_level or level == SilenceLevel.NONE:
            return None

        self._last_silence_level = level
        prompt = self.silence.get_prompt(level)

        if level == SilenceLevel.GOODBYE:
            return {"action": "end_call", "prompt": prompt, "reason": "silence_timeout"}

        return {"action": "silence_prompt", "level": level.value, "prompt": prompt}

    def on_filler_detected(self, filler: str) -> dict[str, Any]:
        """Contact said a filler word — extend endpointing."""
        self._filler_detected = True
        return {
            "action": "extend_endpointing",
            "filler": filler,
            "new_endpointing_ms": self.endpointing.after_filler_ms,
        }

    # ── Backchannel logic ─────────────────────────────────────

    def _should_backchannel(self) -> bool:
        if not self.backchannel.enabled:
            return False
        if self._floor != FloorHolder.TRANSITIONING:
            return False

        now = time.monotonic()
        if (now - self._last_backchannel) * 1000 < self.backchannel.min_interval_ms:
            return False

        # Backchannel if contact has said enough words
        return self._contact_word_count >= self.backchannel.trigger_after_words

    def _pick_backchannel(self) -> str:
        self._last_backchannel = time.monotonic()
        idx = int(self._last_backchannel * 10) % len(self.backchannel.acknowledgment)

        if self._phase == ConversationPhase.NEGOTIATING:
            return self.backchannel.get_response("empathy", idx)
        return self.backchannel.get_response("acknowledgment", idx)

    # ── Interruption strategy ─────────────────────────────────

    def _resolve_interruption_strategy(self) -> InterruptionStrategy:
        if self._ignored_interruptions >= self.interruption.max_ignored_interruptions:
            return InterruptionStrategy.IMMEDIATE_YIELD
        return self.interruption.default_strategy

    # ── Computed endpointing for current context ──────────────

    def get_current_endpointing_ms(self) -> int:
        context = {
            "agent_asked_question": self._agent_asked_question,
            "filler_detected": self._filler_detected,
        }
        return self.endpointing.compute(context)

    # ── State snapshot ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "floor": self._floor.value,
            "phase": self._phase.value,
            "silence_seconds": round(time.monotonic() - self._silence_start, 1) if self._silence_start else 0,
            "silence_level": self._last_silence_level.value,
            "contact_word_count": self._contact_word_count,
            "agent_asked_question": self._agent_asked_question,
            "filler_detected": self._filler_detected,
            "endpointing_ms": self.get_current_endpointing_ms(),
        }
