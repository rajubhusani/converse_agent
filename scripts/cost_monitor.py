#!/usr/bin/env python3
"""
Cost Monitor — Estimate per-minute voice costs from call data.

Reads call history from the database and estimates actual ₹/min costs
based on provider billing rates.

Usage:
    python scripts/cost_monitor.py                  # Summary
    python scripts/cost_monitor.py --last 24h       # Last 24 hours
    python scripts/cost_monitor.py --detailed       # Per-call breakdown
"""
import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# ── Provider rate cards (₹/min) ─────────────────────────────
RATES = {
    "twilio": {
        "telephony": 1.70,        # India outbound PSTN
        "description": "Twilio India outbound",
    },
    "exotel": {
        "telephony": 0.60,        # Exotel India-to-India
        "description": "Exotel India domestic",
    },
    "plivo": {
        "telephony": 0.80,        # Plivo India outbound
        "description": "Plivo India outbound",
    },
    # AI service costs (per minute of conversation)
    "deepgram_nova2": 0.31,       # ~$0.0036/min at Growth tier
    "cartesia_sonic": 1.19,       # ~$0.014/min (40% speech ratio)
    "elevenlabs_turbo": 2.00,     # ~$0.024/min (40% speech ratio)
    "haiku_cached": 0.17,         # ~$0.002/min with prompt caching
    "sonnet": 0.85,               # ~$0.01/min
    "aws_bootstrap": 0.43,        # $0.005/min infrastructure
}


@dataclass
class CallCostEstimate:
    call_id: str
    duration_min: float
    telephony_provider: str
    tts_provider: str
    llm_model: str
    telephony_cost: float
    stt_cost: float
    tts_cost: float
    llm_cost: float
    infra_cost: float

    @property
    def total_cost(self) -> float:
        return self.telephony_cost + self.stt_cost + self.tts_cost + self.llm_cost + self.infra_cost

    @property
    def cost_per_min(self) -> float:
        return self.total_cost / max(self.duration_min, 0.01)


def estimate_call_cost(
    duration_seconds: int,
    telephony_provider: str = "twilio",
    tts_provider: str = "cartesia",
    llm_model: str = "haiku",
    call_id: str = "",
) -> CallCostEstimate:
    """Estimate cost for a single call."""
    duration_min = duration_seconds / 60.0

    # Telephony
    tel_rate = RATES.get(telephony_provider, {}).get("telephony", 1.70)
    telephony_cost = duration_min * tel_rate

    # STT (always Deepgram Nova-2)
    stt_cost = duration_min * RATES["deepgram_nova2"]

    # TTS
    if tts_provider == "cartesia":
        tts_cost = duration_min * RATES["cartesia_sonic"]
    else:
        tts_cost = duration_min * RATES["elevenlabs_turbo"]

    # LLM
    if "haiku" in llm_model:
        llm_cost = duration_min * RATES["haiku_cached"]
    else:
        llm_cost = duration_min * RATES["sonnet"]

    # Infrastructure
    infra_cost = duration_min * RATES["aws_bootstrap"]

    return CallCostEstimate(
        call_id=call_id,
        duration_min=duration_min,
        telephony_provider=telephony_provider,
        tts_provider=tts_provider,
        llm_model=llm_model,
        telephony_cost=telephony_cost,
        stt_cost=stt_cost,
        tts_cost=tts_cost,
        llm_cost=llm_cost,
        infra_cost=infra_cost,
    )


def print_rate_card():
    """Print current rate card comparison."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Voice Stack Cost Comparison (₹/min)                    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║                                                          ║")
    print("║  Component       │ Current Stack │ Optimized Stack       ║")
    print("║  ─────────────── │ ───────────── │ ─────────────────     ║")
    print("║  Telephony       │ Twilio ₹1.70  │ Exotel ₹0.60        ║")
    print("║  STT (Deepgram)  │ ₹0.31         │ ₹0.31               ║")
    print("║  TTS             │ 11Labs ₹2.00  │ Cartesia ₹1.19      ║")
    print("║  LLM             │ Sonnet ₹0.85  │ Haiku+cache ₹0.17   ║")
    print("║  AWS Infra       │ ₹0.43         │ ₹0.43               ║")
    print("║  ─────────────── │ ───────────── │ ─────────────────     ║")
    print("║  TOTAL           │ ₹5.29/min     │ ₹2.70/min           ║")
    print("║                  │               │ (Target: ≤₹4.00)     ║")
    print("║                                                          ║")
    print("╚══════════════════════════════════════════════════════════╝")


def print_scenario_estimates():
    """Show cost estimates for common call scenarios."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Scenario Cost Estimates                                 ║")
    print("╠══════════════════════════════════════════════════════════╣")

    scenarios = [
        ("Quick follow-up (2 min)", 120, "twilio", "cartesia", "haiku"),
        ("Quick follow-up (2 min, Exotel)", 120, "exotel", "cartesia", "haiku"),
        ("Payment call (4 min)", 240, "twilio", "cartesia", "haiku"),
        ("Payment call (4 min, Exotel)", 240, "exotel", "cartesia", "haiku"),
        ("QBR call (10 min)", 600, "twilio", "cartesia", "haiku"),
        ("QBR call (10 min, Exotel)", 600, "exotel", "cartesia", "haiku"),
        ("Legacy stack (4 min)", 240, "twilio", "elevenlabs", "sonnet"),
    ]

    for name, duration, tel, tts, llm in scenarios:
        est = estimate_call_cost(duration, tel, tts, llm)
        print(f"║  {name:<40} ₹{est.total_cost:>6.2f}  ({est.cost_per_min:.2f}/min) ║")

    print("╚══════════════════════════════════════════════════════════╝")

    # Monthly projections
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Monthly Projections (avg 3 min/call)                    ║")
    print("╠══════════════════════════════════════════════════════════╣")

    for calls_per_day in [50, 100, 200, 500]:
        monthly_calls = calls_per_day * 26  # 26 working days
        monthly_min = monthly_calls * 3

        current = estimate_call_cost(180, "twilio", "elevenlabs", "sonnet")
        optimized = estimate_call_cost(180, "exotel", "cartesia", "haiku")

        current_monthly = current.total_cost * monthly_calls
        optimized_monthly = optimized.total_cost * monthly_calls
        savings = current_monthly - optimized_monthly

        print(f"║  {calls_per_day} calls/day ({monthly_calls}/mo):  "
              f"₹{current_monthly:>8,.0f} → ₹{optimized_monthly:>8,.0f}  "
              f"(save ₹{savings:>7,.0f})  ║")

    print("╚══════════════════════════════════════════════════════════╝")


def main():
    parser = argparse.ArgumentParser(description="Voice cost monitoring")
    parser.add_argument("--rates", action="store_true", help="Show rate card comparison")
    parser.add_argument("--estimate", action="store_true", help="Show scenario estimates")
    parser.add_argument("--duration", type=int, default=180, help="Call duration in seconds")
    parser.add_argument("--provider", default="twilio", help="Telephony provider")
    parser.add_argument("--tts", default="cartesia", help="TTS provider")
    parser.add_argument("--llm", default="haiku", help="LLM model")
    args = parser.parse_args()

    if args.rates:
        print_rate_card()
        return

    if args.estimate:
        print_scenario_estimates()
        return

    # Single call estimate
    est = estimate_call_cost(args.duration, args.provider, args.tts, args.llm)
    print(f"\nCall cost estimate ({args.duration}s = {est.duration_min:.1f} min):")
    print(f"  Telephony ({est.telephony_provider}):  ₹{est.telephony_cost:.2f}")
    print(f"  STT (Deepgram):            ₹{est.stt_cost:.2f}")
    print(f"  TTS ({est.tts_provider}):          ₹{est.tts_cost:.2f}")
    print(f"  LLM ({est.llm_model}):            ₹{est.llm_cost:.2f}")
    print(f"  Infrastructure:            ₹{est.infra_cost:.2f}")
    print(f"  ──────────────────────────────")
    print(f"  TOTAL:                     ₹{est.total_cost:.2f}  (₹{est.cost_per_min:.2f}/min)")


if __name__ == "__main__":
    main()
