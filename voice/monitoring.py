"""
Voice monitoring — metrics publishing and health checks.

Publishes per-call and aggregate latency metrics to CloudWatch (or stdout in dev).
Provides health check data for the voice subsystem.
"""
from __future__ import annotations

import asyncio
import structlog
from typing import Any, Optional
from datetime import datetime, timezone

from voice.latency import AggregateLatencyTracker

logger = structlog.get_logger()


class VoiceMetricsPublisher:
    """
    Publishes voice pipeline metrics periodically.

    In production: sends to CloudWatch custom metrics.
    In development: logs to stdout via structlog.
    """

    def __init__(
        self,
        tracker: AggregateLatencyTracker,
        publish_interval_s: int = 30,
        cloudwatch_namespace: str = "ConverseAgent/Voice",
        use_cloudwatch: bool = False,
        aws_region: str = "ap-south-1",
    ):
        self.tracker = tracker
        self.publish_interval_s = publish_interval_s
        self.namespace = cloudwatch_namespace
        self.use_cloudwatch = use_cloudwatch
        self.aws_region = aws_region
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._cw_client = None

    async def start(self) -> None:
        self._running = True
        if self.use_cloudwatch:
            try:
                import boto3
                self._cw_client = boto3.client("cloudwatch", region_name=self.aws_region)
            except ImportError:
                logger.warning("boto3_not_available_using_stdout")
                self.use_cloudwatch = False

        self._task = asyncio.create_task(self._publish_loop(), name="metrics_publisher")
        logger.info("metrics_publisher_started", interval=self.publish_interval_s)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _publish_loop(self) -> None:
        while self._running:
            try:
                await self._publish()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("metrics_publish_error", error=str(e))
            await asyncio.sleep(self.publish_interval_s)

    async def _publish(self) -> None:
        stats = self.tracker.get_all_stats()
        if not stats:
            return

        metrics = self._build_metric_data(stats)

        if self.use_cloudwatch and self._cw_client:
            await asyncio.to_thread(
                self._cw_client.put_metric_data,
                Namespace=self.namespace,
                MetricData=metrics,
            )
        else:
            # Dev mode: log metrics
            for m in metrics:
                logger.info(
                    "voice_metric",
                    name=m["MetricName"],
                    value=m["Value"],
                    unit=m["Unit"],
                )

    def _build_metric_data(self, stats: dict[str, Any]) -> list[dict]:
        """Convert AggregateLatencyTracker stats to CloudWatch MetricData format."""
        now = datetime.now(timezone.utc)
        metrics = []

        # Active calls gauge
        metrics.append({
            "MetricName": "ActiveCalls",
            "Value": stats.get("active_calls", 0),
            "Unit": "Count",
            "Timestamp": now,
        })

        # Per-stage latency percentiles
        for stage in ["stt", "llm_ttfb", "tts_ttfb", "total"]:
            stage_stats = stats.get(stage, {})
            for percentile in ["p50_ms", "p90_ms", "p99_ms"]:
                value = stage_stats.get(percentile, 0)
                if value > 0:
                    metrics.append({
                        "MetricName": f"Pipeline_{stage}_{percentile}",
                        "Value": value,
                        "Unit": "Milliseconds",
                        "Timestamp": now,
                    })

        return metrics

    def get_health(self) -> dict[str, Any]:
        """Health check data for the voice subsystem."""
        stats = self.tracker.get_all_stats()
        total_p90 = stats.get("total", {}).get("p90_ms", 0)
        active = stats.get("active_calls", 0)

        return {
            "status": "healthy" if total_p90 < 800 else "degraded",
            "active_calls": active,
            "latency_p90_ms": total_p90,
            "provider_health": {
                "stt_p90_ms": stats.get("stt", {}).get("p90_ms", 0),
                "llm_p90_ms": stats.get("llm_ttfb", {}).get("p90_ms", 0),
                "tts_p90_ms": stats.get("tts_ttfb", {}).get("p90_ms", 0),
            },
        }


class CostTracker:
    """
    Tracks per-call cost for the ₹4/min budget.

    Usage:
        tracker = CostTracker()
        tracker.record_call(duration_s=180, stt_minutes=3, tts_characters=1125, llm_tokens=3000)
        print(tracker.get_summary())
    """

    # Provider rates (₹ per unit)
    RATES = {
        "exotel_per_min": 0.65,           # ₹/min outbound mobile
        "deepgram_per_min": 0.31,          # ₹/min (Deepgram Growth plan)
        "cartesia_per_1k_chars": 1.19,     # ₹/1000 characters
        "haiku_input_per_mtok": 85.0,      # ₹/MTok ($1 * 85 INR)
        "haiku_output_per_mtok": 425.0,    # ₹/MTok ($5 * 85 INR)
    }

    def __init__(self):
        self._calls: list[dict] = []

    def record_call(
        self,
        call_id: str,
        duration_s: float,
        stt_minutes: float,
        tts_characters: int,
        llm_input_tokens: int,
        llm_output_tokens: int,
    ) -> dict[str, float]:
        """Record a completed call's cost breakdown."""
        duration_min = duration_s / 60

        cost = {
            "call_id": call_id,
            "duration_min": duration_min,
            "telephony_inr": duration_min * self.RATES["exotel_per_min"],
            "stt_inr": stt_minutes * self.RATES["deepgram_per_min"],
            "tts_inr": (tts_characters / 1000) * self.RATES["cartesia_per_1k_chars"],
            "llm_input_inr": (llm_input_tokens / 1_000_000) * self.RATES["haiku_input_per_mtok"],
            "llm_output_inr": (llm_output_tokens / 1_000_000) * self.RATES["haiku_output_per_mtok"],
        }
        cost["total_inr"] = sum(v for k, v in cost.items() if k.endswith("_inr"))
        cost["per_min_inr"] = cost["total_inr"] / duration_min if duration_min > 0 else 0

        self._calls.append(cost)

        logger.info(
            "call_cost",
            call_id=call_id,
            duration_min=round(duration_min, 1),
            total_inr=round(cost["total_inr"], 2),
            per_min_inr=round(cost["per_min_inr"], 2),
        )

        return cost

    def get_summary(self) -> dict[str, Any]:
        """Aggregate cost summary across all tracked calls."""
        if not self._calls:
            return {"total_calls": 0}

        total_inr = sum(c["total_inr"] for c in self._calls)
        total_min = sum(c["duration_min"] for c in self._calls)

        return {
            "total_calls": len(self._calls),
            "total_minutes": round(total_min, 1),
            "total_cost_inr": round(total_inr, 2),
            "avg_per_min_inr": round(total_inr / total_min, 2) if total_min > 0 else 0,
            "breakdown_inr": {
                "telephony": round(sum(c["telephony_inr"] for c in self._calls), 2),
                "stt": round(sum(c["stt_inr"] for c in self._calls), 2),
                "tts": round(sum(c["tts_inr"] for c in self._calls), 2),
                "llm": round(sum(c["llm_input_inr"] + c["llm_output_inr"] for c in self._calls), 2),
            },
        }
