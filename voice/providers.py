"""
Voice Providers — Multi-provider STT and TTS configuration with latency profiles.

Supports configurable provider switching for both STT and TTS with
provider-specific parameter mappings, quality presets, and streaming
settings optimized for low-latency voice conversations.

STT Providers: Deepgram, AssemblyAI, Google Cloud, Azure, Whisper (local)
TTS Providers: ElevenLabs, Cartesia, PlayHT, Azure, Google Cloud, XTTS (local)
Telephony:     Twilio, Vonage, Telnyx, Daily (WebRTC)

Each provider carries latency characteristics so the pipeline builder
can make informed tradeoffs between latency and quality.
"""
from __future__ import annotations

import structlog
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
#  ENUMS
# ══════════════════════════════════════════════════════════════

class STTProvider(str, Enum):
    DEEPGRAM = "deepgram"
    ASSEMBLYAI = "assemblyai"
    GOOGLE = "google"
    AZURE = "azure"
    WHISPER = "whisper"         # local / OpenAI API


class TTSProvider(str, Enum):
    ELEVENLABS = "elevenlabs"
    CARTESIA = "cartesia"       # sonic — sub-100ms TTFB
    PLAYHT = "playht"
    AZURE = "azure"
    GOOGLE = "google"
    XTTS = "xtts"              # local coqui model


class TelephonyProvider(str, Enum):
    TWILIO = "twilio"
    VONAGE = "vonage"
    TELNYX = "telnyx"
    DAILY = "daily"            # WebRTC only (no PSTN)
    EXOTEL = "exotel"          # India — ₹0.50-0.70/min
    PLIVO = "plivo"            # India/Global — ₹0.60-0.80/min
    OZONETEL = "ozonetel"      # India — ₹0.55-0.75/min


class AudioCodec(str, Enum):
    PCM_16K = "pcm_16000"      # 16-bit PCM at 16kHz — lowest latency
    PCM_24K = "pcm_24000"      # 16-bit PCM at 24kHz — better quality
    MULAW_8K = "mulaw_8000"    # μ-law at 8kHz — telephony native
    OPUS = "opus"              # variable bitrate — WebRTC
    MP3 = "mp3"                # compressed — higher latency


class QualityPreset(str, Enum):
    ULTRA_LOW_LATENCY = "ultra_low_latency"   # <300ms total pipeline
    LOW_LATENCY = "low_latency"               # <500ms total pipeline
    BALANCED = "balanced"                      # <800ms, better quality
    HIGH_QUALITY = "high_quality"              # best quality, ~1s+
    COST_OPTIMIZED = "cost_optimized"          # ₹4/min target: Haiku + Cartesia


# ══════════════════════════════════════════════════════════════
#  LATENCY PROFILES
# ══════════════════════════════════════════════════════════════

@dataclass
class LatencyProfile:
    """Expected latency characteristics for a provider configuration."""
    time_to_first_byte_ms: int        # TTFB for streaming response
    processing_per_second_ms: int     # ms per second of audio processed
    warmup_ms: int = 0                # cold start penalty
    streaming: bool = True            # supports chunked streaming?
    websocket: bool = True            # supports persistent WS connection?

    @property
    def estimated_round_trip_ms(self) -> int:
        """Estimated latency for a typical 3-second utterance."""
        return self.time_to_first_byte_ms + (self.processing_per_second_ms * 3)


# ══════════════════════════════════════════════════════════════
#  STT PROVIDER CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class STTConfig:
    """Complete configuration for a speech-to-text provider."""
    provider: STTProvider
    api_key: str = ""
    model: str = ""
    language: str = "en"

    # Streaming & real-time
    interim_results: bool = True         # stream partial transcripts
    endpointing_ms: int = 250           # silence before finalizing (lower = faster)
    vad_enabled: bool = True             # voice activity detection
    vad_threshold: float = 0.5           # sensitivity (0-1, higher = more sensitive)
    utterance_end_ms: int = 1000         # max pause within a single utterance

    # Quality
    smart_format: bool = True            # auto-punctuate, format numbers
    profanity_filter: bool = False
    diarize: bool = False                # speaker identification
    keywords: list[str] = field(default_factory=list)  # boost recognition
    keyword_boost: float = 1.5

    # Audio input
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "linear16"

    # Noise & environment
    noise_reduction: bool = True
    multichannel: bool = False           # separate agent/contact streams

    latency: LatencyProfile = field(default_factory=lambda: LatencyProfile(
        time_to_first_byte_ms=150, processing_per_second_ms=50
    ))

    def to_provider_params(self) -> dict[str, Any]:
        """Convert to provider-specific API parameters."""
        if self.provider == STTProvider.DEEPGRAM:
            return self._deepgram_params()
        elif self.provider == STTProvider.ASSEMBLYAI:
            return self._assemblyai_params()
        elif self.provider == STTProvider.GOOGLE:
            return self._google_params()
        elif self.provider == STTProvider.AZURE:
            return self._azure_params()
        elif self.provider == STTProvider.WHISPER:
            return self._whisper_params()
        return {}

    def _deepgram_params(self) -> dict[str, Any]:
        params = {
            "model": self.model or "nova-2",
            "language": self.language,
            "interim_results": self.interim_results,
            "endpointing": self.endpointing_ms,
            "vad_events": self.vad_enabled,
            "smart_format": self.smart_format,
            "punctuate": True,
            "filler_words": True,          # detect "um", "uh" for naturalness
            "utterance_end_ms": str(self.utterance_end_ms),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "encoding": self.encoding,
        }
        if self.keywords:
            params["keywords"] = [f"{k}:{self.keyword_boost}" for k in self.keywords]
        if self.multichannel:
            params["multichannel"] = True
        if self.diarize:
            params["diarize"] = True
        return params

    def _assemblyai_params(self) -> dict[str, Any]:
        return {
            "speech_model": self.model or "best",
            "language_code": self.language,
            "word_boost": self.keywords,
            "boost_param": "high" if self.keyword_boost > 1.5 else "default",
            "format_text": self.smart_format,
            "punctuate": True,
            "filter_profanity": self.profanity_filter,
            "sample_rate": self.sample_rate,
        }

    def _google_params(self) -> dict[str, Any]:
        return {
            "model": self.model or "latest_long",
            "language_code": self.language,
            "enable_automatic_punctuation": self.smart_format,
            "interim_results": self.interim_results,
            "single_utterance": False,
            "sample_rate_hertz": self.sample_rate,
            "encoding": "LINEAR16",
            "speech_contexts": [{"phrases": self.keywords, "boost": int(self.keyword_boost * 10)}]
            if self.keywords else [],
        }

    def _azure_params(self) -> dict[str, Any]:
        return {
            "speech_recognition_language": self.language,
            "output_format": "detailed",
            "profanity_option": "masked" if self.profanity_filter else "raw",
            "enable_dictation": False,
            "initial_silence_timeout_ms": str(self.endpointing_ms * 4),
            "end_silence_timeout_ms": str(self.endpointing_ms),
        }

    def _whisper_params(self) -> dict[str, Any]:
        return {
            "model": self.model or "whisper-1",
            "language": self.language[:2],
            "response_format": "verbose_json",
            "temperature": 0.0,
        }


# ══════════════════════════════════════════════════════════════
#  TTS PROVIDER CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class TTSConfig:
    """Complete configuration for a text-to-speech provider."""
    provider: TTSProvider
    api_key: str = ""
    voice_id: str = ""
    model: str = ""

    # Voice characteristics
    stability: float = 0.5              # 0-1: lower = more expressive
    similarity_boost: float = 0.75      # 0-1: voice clone fidelity
    speed: float = 1.0                  # 0.5-2.0: speech rate
    pitch: float = 0.0                  # semitones adjustment

    # Streaming & format
    output_format: AudioCodec = AudioCodec.PCM_16K
    sample_rate: int = 16000
    optimize_streaming_latency: int = 4  # ElevenLabs: 0-4 (4=lowest latency)
    chunk_length_schedule: list[int] = field(default_factory=lambda: [50])  # ms per chunk

    # Naturalness
    style_exaggeration: float = 0.0     # 0-1: emotional intensity
    speaker_boost: bool = True          # enhance clarity
    use_ssml: bool = False              # SSML markup support

    latency: LatencyProfile = field(default_factory=lambda: LatencyProfile(
        time_to_first_byte_ms=200, processing_per_second_ms=30
    ))

    def to_provider_params(self) -> dict[str, Any]:
        if self.provider == TTSProvider.ELEVENLABS:
            return self._elevenlabs_params()
        elif self.provider == TTSProvider.CARTESIA:
            return self._cartesia_params()
        elif self.provider == TTSProvider.PLAYHT:
            return self._playht_params()
        elif self.provider == TTSProvider.AZURE:
            return self._azure_params()
        elif self.provider == TTSProvider.GOOGLE:
            return self._google_params()
        elif self.provider == TTSProvider.XTTS:
            return self._xtts_params()
        return {}

    def _elevenlabs_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model or "eleven_turbo_v2_5",
            "voice_id": self.voice_id or "21m00Tcm4TlvDq8ikWAM",
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style_exaggeration,
                "use_speaker_boost": self.speaker_boost,
            },
            "output_format": self._elevenlabs_format(),
            "optimize_streaming_latency": self.optimize_streaming_latency,
        }

    def _elevenlabs_format(self) -> str:
        fmt_map = {
            AudioCodec.PCM_16K: "pcm_16000",
            AudioCodec.PCM_24K: "pcm_24000",
            AudioCodec.MULAW_8K: "ulaw_8000",
            AudioCodec.MP3: "mp3_44100_128",
        }
        return fmt_map.get(self.output_format, "pcm_16000")

    def _cartesia_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model or "sonic-english",
            "voice": {"mode": "id", "id": self.voice_id or "a0e99841-438c-4a64-b679-ae501e7d6091"},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "language": "en",
            "speed": "" if self.speed == 1.0 else ("slow" if self.speed < 0.9 else "fast"),
        }

    def _playht_params(self) -> dict[str, Any]:
        return {
            "voice": self.voice_id,
            "output_format": "raw",
            "sample_rate": self.sample_rate,
            "speed": self.speed,
            "quality": "draft" if self.optimize_streaming_latency > 2 else "medium",
        }

    def _azure_params(self) -> dict[str, Any]:
        return {
            "voice_name": self.voice_id or "en-US-JennyNeural",
            "output_format": "Raw16Khz16BitMonoPcm",
            "speaking_rate": f"{int(self.speed * 100)}%",
            "pitch": f"{self.pitch:+.0f}st" if self.pitch else "default",
        }

    def _google_params(self) -> dict[str, Any]:
        return {
            "voice": {"language_code": "en-US", "name": self.voice_id or "en-US-Neural2-F"},
            "audio_config": {
                "audio_encoding": "LINEAR16",
                "sample_rate_hertz": self.sample_rate,
                "speaking_rate": self.speed,
                "pitch": self.pitch,
            },
        }

    def _xtts_params(self) -> dict[str, Any]:
        return {
            "model": self.model or "xtts_v2",
            "speaker_wav": self.voice_id,  # path to reference audio
            "language": "en",
            "temperature": 0.7,
            "speed": self.speed,
            "stream": True,
        }


# ══════════════════════════════════════════════════════════════
#  TELEPHONY CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class TelephonyConfig:
    """Telephony provider configuration."""
    provider: TelephonyProvider
    account_sid: str = ""
    api_key: str = ""                          # Exotel: separate API key for Basic Auth
    auth_token: str = ""
    phone_number: str = ""
    sip_domain: str = ""
    subdomain: str = ""                        # Exotel: API subdomain (api.exotel.com)
    status_callback_url: str = ""
    ring_timeout_s: int = 30
    answering_machine_detection: bool = True
    amd_timeout_ms: int = 3000             # how long to wait for AMD result
    recording_enabled: bool = False
    codec: AudioCodec = AudioCodec.MULAW_8K  # telephony native


# ══════════════════════════════════════════════════════════════
#  QUALITY PRESETS
# ══════════════════════════════════════════════════════════════

# Pre-built configurations optimized for different latency/quality tradeoffs
QUALITY_PRESETS: dict[QualityPreset, dict[str, Any]] = {
    QualityPreset.ULTRA_LOW_LATENCY: {
        "stt": {
            "provider": STTProvider.DEEPGRAM,
            "model": "nova-2",
            "endpointing_ms": 150,           # aggressive endpointing
            "vad_threshold": 0.6,
            "utterance_end_ms": 600,
            "latency": LatencyProfile(time_to_first_byte_ms=80, processing_per_second_ms=30, websocket=True),
        },
        "tts": {
            "provider": TTSProvider.CARTESIA,
            "model": "sonic-english",
            "optimize_streaming_latency": 4,
            "output_format": AudioCodec.PCM_16K,
            "chunk_length_schedule": [30],
            "latency": LatencyProfile(time_to_first_byte_ms=60, processing_per_second_ms=20, websocket=True),
        },
        "llm": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 150,
            "temperature": 0.7,
        },
        "pipeline": {
            "target_latency_ms": 300,
            "llm_sentence_streaming": True,    # start TTS on first sentence
            "prefetch_greeting": True,         # pre-generate greeting before call connects
        },
    },
    QualityPreset.LOW_LATENCY: {
        "stt": {
            "provider": STTProvider.DEEPGRAM,
            "model": "nova-2",
            "endpointing_ms": 250,
            "vad_threshold": 0.5,
            "utterance_end_ms": 800,
            "latency": LatencyProfile(time_to_first_byte_ms=100, processing_per_second_ms=40, websocket=True),
        },
        "tts": {
            "provider": TTSProvider.ELEVENLABS,
            "model": "eleven_turbo_v2_5",
            "optimize_streaming_latency": 4,
            "output_format": AudioCodec.PCM_16K,
            "chunk_length_schedule": [50],
            "latency": LatencyProfile(time_to_first_byte_ms=150, processing_per_second_ms=30, websocket=True),
        },
        "llm": {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 200,
            "temperature": 0.7,
        },
        "pipeline": {
            "target_latency_ms": 500,
            "llm_sentence_streaming": True,
            "prefetch_greeting": True,
        },
    },
    QualityPreset.BALANCED: {
        "stt": {
            "provider": STTProvider.DEEPGRAM,
            "model": "nova-2",
            "endpointing_ms": 350,
            "vad_threshold": 0.45,
            "utterance_end_ms": 1000,
            "latency": LatencyProfile(time_to_first_byte_ms=100, processing_per_second_ms=40, websocket=True),
        },
        "tts": {
            "provider": TTSProvider.ELEVENLABS,
            "model": "eleven_multilingual_v2",
            "optimize_streaming_latency": 2,
            "output_format": AudioCodec.PCM_24K,
            "chunk_length_schedule": [100],
            "latency": LatencyProfile(time_to_first_byte_ms=250, processing_per_second_ms=40, websocket=True),
        },
        "llm": {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 300,
            "temperature": 0.7,
        },
        "pipeline": {
            "target_latency_ms": 800,
            "llm_sentence_streaming": True,
            "prefetch_greeting": False,
        },
    },
    QualityPreset.HIGH_QUALITY: {
        "stt": {
            "provider": STTProvider.DEEPGRAM,
            "model": "nova-2",
            "endpointing_ms": 500,
            "vad_threshold": 0.4,
            "utterance_end_ms": 1200,
            "diarize": True,
            "latency": LatencyProfile(time_to_first_byte_ms=120, processing_per_second_ms=50, websocket=True),
        },
        "tts": {
            "provider": TTSProvider.ELEVENLABS,
            "model": "eleven_multilingual_v2",
            "optimize_streaming_latency": 0,
            "output_format": AudioCodec.PCM_24K,
            "chunk_length_schedule": [200],
            "latency": LatencyProfile(time_to_first_byte_ms=400, processing_per_second_ms=50, websocket=True),
        },
        "llm": {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 500,
            "temperature": 0.7,
        },
        "pipeline": {
            "target_latency_ms": 1200,
            "llm_sentence_streaming": False,
            "prefetch_greeting": False,
        },
    },
    QualityPreset.COST_OPTIMIZED: {
        "stt": {
            "provider": STTProvider.DEEPGRAM,
            "model": "nova-2",
            "endpointing_ms": 200,
            "vad_threshold": 0.55,
            "utterance_end_ms": 700,
            "latency": LatencyProfile(time_to_first_byte_ms=80, processing_per_second_ms=30, websocket=True),
        },
        "tts": {
            "provider": TTSProvider.CARTESIA,
            "model": "sonic-english",
            "optimize_streaming_latency": 4,
            "output_format": AudioCodec.PCM_16K,
            "chunk_length_schedule": [30],
            "latency": LatencyProfile(time_to_first_byte_ms=60, processing_per_second_ms=20, websocket=True),
        },
        "llm": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 150,
            "temperature": 0.7,
        },
        "pipeline": {
            "target_latency_ms": 400,
            "llm_sentence_streaming": True,
            "prefetch_greeting": True,
        },
    },
}


# ══════════════════════════════════════════════════════════════
#  PROVIDER REGISTRY
# ══════════════════════════════════════════════════════════════

class VoiceProviderRegistry:
    """
    Central registry for voice provider configurations.

    Manages STT, TTS, and telephony configs. Supports multiple named
    configurations per provider type for A/B testing or per-contact
    provider selection.
    """

    def __init__(self):
        self._stt_configs: dict[str, STTConfig] = {}
        self._tts_configs: dict[str, TTSConfig] = {}
        self._telephony_config: Optional[TelephonyConfig] = None
        self._default_stt: str = "default"
        self._default_tts: str = "default"

    # ── Registration ──────────────────────────────────────────

    def register_stt(self, name: str, config: STTConfig, default: bool = False) -> None:
        self._stt_configs[name] = config
        if default or not self._stt_configs:
            self._default_stt = name
        logger.info("stt_registered", name=name, provider=config.provider.value)

    def register_tts(self, name: str, config: TTSConfig, default: bool = False) -> None:
        self._tts_configs[name] = config
        if default or not self._tts_configs:
            self._default_tts = name
        logger.info("tts_registered", name=name, provider=config.provider.value)

    def set_telephony(self, config: TelephonyConfig) -> None:
        self._telephony_config = config
        logger.info("telephony_configured", provider=config.provider.value)

    # ── Lookup ────────────────────────────────────────────────

    def get_stt(self, name: str = "") -> Optional[STTConfig]:
        return self._stt_configs.get(name or self._default_stt)

    def get_tts(self, name: str = "") -> Optional[TTSConfig]:
        return self._tts_configs.get(name or self._default_tts)

    def get_telephony(self) -> Optional[TelephonyConfig]:
        return self._telephony_config

    # ── Preset loading ────────────────────────────────────────

    def load_preset(self, preset: QualityPreset, api_keys: dict[str, str] = None) -> None:
        """Load a complete quality preset, optionally injecting API keys."""
        api_keys = api_keys or {}
        config = QUALITY_PRESETS[preset]

        stt_data = config["stt"]
        stt_config = STTConfig(
            provider=stt_data["provider"],
            api_key=api_keys.get("stt", ""),
            model=stt_data.get("model", ""),
            endpointing_ms=stt_data.get("endpointing_ms", 250),
            vad_threshold=stt_data.get("vad_threshold", 0.5),
            utterance_end_ms=stt_data.get("utterance_end_ms", 1000),
            diarize=stt_data.get("diarize", False),
            latency=stt_data.get("latency", LatencyProfile(100, 40)),
        )
        self.register_stt("default", stt_config, default=True)

        tts_data = config["tts"]
        tts_config = TTSConfig(
            provider=tts_data["provider"],
            api_key=api_keys.get("tts", ""),
            model=tts_data.get("model", ""),
            optimize_streaming_latency=tts_data.get("optimize_streaming_latency", 2),
            output_format=tts_data.get("output_format", AudioCodec.PCM_16K),
            chunk_length_schedule=tts_data.get("chunk_length_schedule", [50]),
            latency=tts_data.get("latency", LatencyProfile(200, 30)),
        )
        self.register_tts("default", tts_config, default=True)

        logger.info("preset_loaded", preset=preset.value)

    # ── Estimated latency ─────────────────────────────────────

    def estimated_pipeline_latency_ms(self, stt_name: str = "", tts_name: str = "") -> int:
        """Estimate total pipeline latency for a given provider combination."""
        stt = self.get_stt(stt_name)
        tts = self.get_tts(tts_name)
        stt_latency = stt.latency.time_to_first_byte_ms if stt else 200
        tts_latency = tts.latency.time_to_first_byte_ms if tts else 300
        llm_latency = 200   # typical streaming TTFB for Claude
        return stt_latency + llm_latency + tts_latency

    def list_stt_providers(self) -> list[dict[str, Any]]:
        return [
            {"name": n, "provider": c.provider.value, "model": c.model,
             "latency_ms": c.latency.time_to_first_byte_ms}
            for n, c in self._stt_configs.items()
        ]

    def list_tts_providers(self) -> list[dict[str, Any]]:
        return [
            {"name": n, "provider": c.provider.value, "model": c.model,
             "voice_id": c.voice_id, "latency_ms": c.latency.time_to_first_byte_ms}
            for n, c in self._tts_configs.items()
        ]
