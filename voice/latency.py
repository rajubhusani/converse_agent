"""
Latency Tracker — Per-stage latency measurement and optimization.

Tracks latency across every stage of the voice pipeline (STT, LLM, TTS,
network) with:
- Per-stage high-resolution timing
- Rolling percentile tracking (p50, p90, p99)
- Latency budget enforcement with violation alerting
- Adaptive optimization hints (when to degrade quality for speed)
- Per-call and aggregate latency reporting
"""
from __future__ import annotations

import time
import structlog
from collections import deque
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = structlog.get_logger()


class PipelineStage(str, Enum):
    """Stages in the voice pipeline, measured independently."""
    STT = "stt"                  # speech-to-text transcription
    STT_ENDPOINTING = "stt_endpointing"  # silence detection → finalize
    LLM_TTFB = "llm_ttfb"       # time to first LLM token
    LLM_FULL = "llm_full"       # full LLM generation
    TTS_TTFB = "tts_ttfb"       # time to first audio byte
    TTS_FULL = "tts_full"       # full audio generation
    NETWORK = "network"          # transport overhead
    TOTAL = "total"              # end-to-end: speech end → audio start


@dataclass
class LatencyBudget:
    """
    Latency budget per stage. When a stage exceeds its budget,
    the tracker emits optimization hints.
    """
    stt_ms: int = 150
    llm_ttfb_ms: int = 250
    tts_ttfb_ms: int = 200
    network_ms: int = 50
    total_ms: int = 600          # target end-to-end

    def budget_for(self, stage: PipelineStage) -> int:
        return {
            PipelineStage.STT: self.stt_ms,
            PipelineStage.LLM_TTFB: self.llm_ttfb_ms,
            PipelineStage.TTS_TTFB: self.tts_ttfb_ms,
            PipelineStage.NETWORK: self.network_ms,
            PipelineStage.TOTAL: self.total_ms,
        }.get(stage, 500)


@dataclass
class LatencyMeasurement:
    """A single latency measurement."""
    stage: PipelineStage
    duration_ms: float
    timestamp: float
    call_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class StageTracker:
    """Tracks latency measurements for a single pipeline stage."""

    def __init__(self, stage: PipelineStage, window_size: int = 200):
        self.stage = stage
        self._measurements: deque[float] = deque(maxlen=window_size)
        self._total: float = 0.0
        self._count: int = 0
        self._min: float = float("inf")
        self._max: float = 0.0

    def record(self, duration_ms: float) -> None:
        self._measurements.append(duration_ms)
        self._total += duration_ms
        self._count += 1
        self._min = min(self._min, duration_ms)
        self._max = max(self._max, duration_ms)

    @property
    def avg_ms(self) -> float:
        return self._total / self._count if self._count > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        return self._percentile(50)

    @property
    def p90_ms(self) -> float:
        return self._percentile(90)

    @property
    def p99_ms(self) -> float:
        return self._percentile(99)

    @property
    def min_ms(self) -> float:
        return self._min if self._count > 0 else 0.0

    @property
    def max_ms(self) -> float:
        return self._max

    @property
    def count(self) -> int:
        return self._count

    def _percentile(self, pct: int) -> float:
        if not self._measurements:
            return 0.0
        sorted_vals = sorted(self._measurements)
        idx = int(len(sorted_vals) * pct / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "count": self._count,
            "avg_ms": round(self.avg_ms, 1),
            "p50_ms": round(self.p50_ms, 1),
            "p90_ms": round(self.p90_ms, 1),
            "p99_ms": round(self.p99_ms, 1),
            "min_ms": round(self.min_ms, 1),
            "max_ms": round(self.max_ms, 1),
        }


# ══════════════════════════════════════════════════════════════
#  CALL LATENCY TRACKER
# ══════════════════════════════════════════════════════════════

class CallLatencyTracker:
    """
    Tracks latency for a single active call.

    Usage:
        tracker.start(PipelineStage.STT)
        # ... STT processes audio ...
        tracker.end(PipelineStage.STT)

        tracker.start(PipelineStage.LLM_TTFB)
        # ... waiting for first LLM token ...
        tracker.end(PipelineStage.LLM_TTFB)
    """

    def __init__(self, call_id: str, budget: LatencyBudget = None):
        self.call_id = call_id
        self.budget = budget or LatencyBudget()
        self._starts: dict[str, float] = {}
        self._measurements: list[LatencyMeasurement] = []
        self._turn_count: int = 0
        self._violations: list[dict[str, Any]] = []

    def start(self, stage: PipelineStage) -> None:
        """Mark the start of a pipeline stage."""
        self._starts[stage.value] = time.monotonic()

    def end(self, stage: PipelineStage, metadata: dict[str, Any] = None) -> float:
        """
        Mark the end of a pipeline stage.
        Returns duration in ms. Records violation if over budget.
        """
        start = self._starts.pop(stage.value, None)
        if start is None:
            return 0.0

        duration_ms = (time.monotonic() - start) * 1000
        measurement = LatencyMeasurement(
            stage=stage,
            duration_ms=duration_ms,
            timestamp=time.monotonic(),
            call_id=self.call_id,
            metadata=metadata or {},
        )
        self._measurements.append(measurement)

        # Budget check
        budget = self.budget.budget_for(stage)
        if duration_ms > budget:
            violation = {
                "stage": stage.value,
                "duration_ms": round(duration_ms, 1),
                "budget_ms": budget,
                "overage_ms": round(duration_ms - budget, 1),
                "turn": self._turn_count,
            }
            self._violations.append(violation)
            logger.warning("latency_budget_exceeded", call_id=self.call_id, **violation)

        return duration_ms

    def record_turn(self) -> None:
        """Mark a complete conversational turn (user spoke → agent replied)."""
        self._turn_count += 1

    def get_last_total_ms(self) -> float:
        """Get the last total pipeline latency measurement."""
        for m in reversed(self._measurements):
            if m.stage == PipelineStage.TOTAL:
                return m.duration_ms
        return 0.0

    def get_optimization_hints(self) -> list[dict[str, Any]]:
        """
        Analyze recent latency and suggest optimizations.
        Returns a list of actionable hints.
        """
        hints = []
        recent = self._measurements[-20:]  # last 20 measurements

        # Check if any stage is consistently over budget
        for stage in PipelineStage:
            stage_measurements = [m.duration_ms for m in recent if m.stage == stage]
            if not stage_measurements:
                continue
            avg = sum(stage_measurements) / len(stage_measurements)
            budget = self.budget.budget_for(stage)

            if avg > budget * 1.5:
                if stage == PipelineStage.LLM_TTFB:
                    hints.append({
                        "stage": stage.value,
                        "hint": "switch_to_faster_model",
                        "reason": f"LLM TTFB avg {avg:.0f}ms exceeds budget {budget}ms",
                        "suggestion": "Consider claude-haiku for lower latency",
                    })
                elif stage == PipelineStage.TTS_TTFB:
                    hints.append({
                        "stage": stage.value,
                        "hint": "increase_streaming_optimization",
                        "reason": f"TTS TTFB avg {avg:.0f}ms exceeds budget {budget}ms",
                        "suggestion": "Increase optimize_streaming_latency or switch to Cartesia",
                    })
                elif stage == PipelineStage.STT:
                    hints.append({
                        "stage": stage.value,
                        "hint": "reduce_endpointing",
                        "reason": f"STT avg {avg:.0f}ms exceeds budget {budget}ms",
                        "suggestion": "Lower endpointing_ms for faster finalization",
                    })

        return hints

    @property
    def violations(self) -> list[dict[str, Any]]:
        return self._violations

    def to_dict(self) -> dict[str, Any]:
        stage_stats = {}
        for stage in PipelineStage:
            durations = [m.duration_ms for m in self._measurements if m.stage == stage]
            if durations:
                stage_stats[stage.value] = {
                    "count": len(durations),
                    "avg_ms": round(sum(durations) / len(durations), 1),
                    "min_ms": round(min(durations), 1),
                    "max_ms": round(max(durations), 1),
                }
        return {
            "call_id": self.call_id,
            "turns": self._turn_count,
            "total_measurements": len(self._measurements),
            "violations": len(self._violations),
            "stages": stage_stats,
        }


# ══════════════════════════════════════════════════════════════
#  AGGREGATE LATENCY TRACKER
# ══════════════════════════════════════════════════════════════

class AggregateLatencyTracker:
    """
    Tracks latency across all calls for system-wide monitoring.
    Maintains per-stage rolling percentiles.
    """

    def __init__(self, budget: LatencyBudget = None):
        self.budget = budget or LatencyBudget()
        self._stages: dict[PipelineStage, StageTracker] = {
            stage: StageTracker(stage) for stage in PipelineStage
        }
        self._call_trackers: dict[str, CallLatencyTracker] = {}

    def create_call_tracker(self, call_id: str) -> CallLatencyTracker:
        tracker = CallLatencyTracker(call_id, self.budget)
        self._call_trackers[call_id] = tracker
        return tracker

    def get_call_tracker(self, call_id: str) -> Optional[CallLatencyTracker]:
        return self._call_trackers.get(call_id)

    def record(self, stage: PipelineStage, duration_ms: float, call_id: str = "") -> None:
        """Record a measurement in the aggregate tracker."""
        self._stages[stage].record(duration_ms)

    def remove_call(self, call_id: str) -> Optional[CallLatencyTracker]:
        return self._call_trackers.pop(call_id, None)

    def get_stage_stats(self, stage: PipelineStage) -> dict[str, Any]:
        return self._stages[stage].to_dict()

    def get_all_stats(self) -> dict[str, Any]:
        stats = {}
        for stage, tracker in self._stages.items():
            if tracker.count > 0:
                stats[stage.value] = tracker.to_dict()
        stats["active_calls"] = len(self._call_trackers)
        stats["budget"] = {
            "total_ms": self.budget.total_ms,
            "stt_ms": self.budget.stt_ms,
            "llm_ttfb_ms": self.budget.llm_ttfb_ms,
            "tts_ttfb_ms": self.budget.tts_ttfb_ms,
        }
        return stats

    def is_within_budget(self, stage: PipelineStage) -> bool:
        """Check if the stage's p90 is within budget."""
        tracker = self._stages[stage]
        return tracker.p90_ms <= self.budget.budget_for(stage)
