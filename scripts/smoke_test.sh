#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  Smoke Test — Validate voice stack configuration
#
#  Run after deploy to verify all services are reachable
#  and APIs return expected responses.
#
#  Usage:
#    bash scripts/smoke_test.sh               # Test local
#    bash scripts/smoke_test.sh https://your-domain.com  # Test remote
# ══════════════════════════════════════════════════════════════
set -euo pipefail

BASE="${1:-http://localhost:8000}"
PASS=0
FAIL=0
WARN=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}✗${NC} $1"; FAIL=$((FAIL + 1)); }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; WARN=$((WARN + 1)); }

echo "╔══════════════════════════════════════════════════╗"
echo "║  ConverseAgent Smoke Test                        ║"
echo "║  Target: $BASE"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Health Check ──────────────────────────────────────────
echo "1. Health Check"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE/health" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    pass "GET /health → 200"
else
    fail "GET /health → $HTTP_CODE (expected 200)"
fi

# ── 2. Voice Endpoints ──────────────────────────────────────
echo "2. Voice API Endpoints"

# List active calls
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE/api/v1/voice/calls" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    pass "GET /api/v1/voice/calls → 200"
else
    fail "GET /api/v1/voice/calls → $HTTP_CODE"
fi

# Voice status endpoint (POST with empty body should return 200)
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
    -X POST -H "Content-Type: application/json" \
    -d '{"call_id":"smoke_test","status":"completed"}' \
    "$BASE/api/v1/voice/status" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    pass "POST /api/v1/voice/status → 200"
else
    fail "POST /api/v1/voice/status → $HTTP_CODE"
fi

# ── 3. Webhook Endpoints ────────────────────────────────────
echo "3. Webhook Endpoints"

# Twilio webhook
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
    -X POST -H "Content-Type: application/x-www-form-urlencoded" \
    -d "CallSid=CA_smoke&CallStatus=completed&From=%2B14155551234&To=%2B919876543210" \
    "$BASE/webhooks/twilio/status" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    pass "POST /webhooks/twilio/status → 200"
else
    fail "POST /webhooks/twilio/status → $HTTP_CODE"
fi

# Twilio answer URL (returns TwiML)
RESPONSE=$(curl -sf -X POST \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "CallSid=CA_smoke_answer" \
    "$BASE/webhooks/twilio/answer" 2>/dev/null || echo "ERROR")
if echo "$RESPONSE" | grep -q "<Sip>"; then
    pass "POST /webhooks/twilio/answer → TwiML with <Sip>"
else
    fail "POST /webhooks/twilio/answer → missing TwiML"
fi

# Exotel webhook
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
    -X POST -H "Content-Type: application/json" \
    -d '{"CallSid":"exo_smoke","Status":"completed"}' \
    "$BASE/webhooks/exotel/status" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    pass "POST /webhooks/exotel/status → 200"
else
    fail "POST /webhooks/exotel/status → $HTTP_CODE"
fi

# Plivo webhook
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
    -X POST -H "Content-Type: application/x-www-form-urlencoded" \
    -d "CallUUID=plivo_smoke&Event=hangup" \
    "$BASE/webhooks/plivo/status" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    pass "POST /webhooks/plivo/status → 200"
else
    fail "POST /webhooks/plivo/status → $HTTP_CODE"
fi

# ── 4. Provider Configuration ───────────────────────────────
echo "4. Provider Configuration"

if [ -f .env ]; then
    source .env 2>/dev/null || true

    # Check Twilio
    if [ -n "${TWILIO_SID:-}" ] && [ "${TWILIO_SID}" != "your_twilio_account_sid" ]; then
        pass "Twilio credentials configured"
    else
        warn "Twilio credentials not set (stub mode)"
    fi

    # Check Deepgram
    if [ -n "${DEEPGRAM_API_KEY:-}" ] && [ "${DEEPGRAM_API_KEY}" != "your_deepgram_api_key" ]; then
        pass "Deepgram API key configured"
    else
        warn "Deepgram API key not set (STT unavailable)"
    fi

    # Check Cartesia
    if [ -n "${CARTESIA_API_KEY:-}" ] && [ "${CARTESIA_API_KEY}" != "your_cartesia_api_key" ]; then
        pass "Cartesia API key configured"
    else
        warn "Cartesia API key not set (TTS unavailable)"
    fi

    # Check Anthropic
    if [ -n "${ANTHROPIC_API_KEY:-}" ] && [[ "${ANTHROPIC_API_KEY}" == sk-ant-* ]]; then
        pass "Anthropic API key configured"
    else
        warn "Anthropic API key not set or invalid"
    fi
else
    warn "No .env file found"
fi

# ── 5. Docker Services ──────────────────────────────────────
echo "5. Docker Services"

COMPOSE_FILE="docker-compose.prod.yml"
if [ -f "$COMPOSE_FILE" ]; then
    for svc in app worker voice redis postgres; do
        STATUS=$(docker compose -f "$COMPOSE_FILE" ps "$svc" --format "{{.Status}}" 2>/dev/null || echo "")
        if echo "$STATUS" | grep -qi "up\|running"; then
            pass "Service '$svc' running"
        elif [ -z "$STATUS" ]; then
            warn "Service '$svc' not found (not deployed yet?)"
        else
            fail "Service '$svc' status: $STATUS"
        fi
    done
else
    warn "docker-compose.prod.yml not found — skipping service checks"
fi

# ── 6. Network / SIP ────────────────────────────────────────
echo "6. Network / SIP"

# Check if SIP port is open
if command -v ss &>/dev/null; then
    if ss -ulnp | grep -q ":5060"; then
        pass "SIP port 5060/udp listening"
    else
        warn "SIP port 5060/udp not listening (voice container may not be running)"
    fi
fi

# Check RTP range
if command -v ss &>/dev/null; then
    RTP_COUNT=$(ss -ulnp | grep -c "1000[0-9]" 2>/dev/null || echo "0")
    if [ "$RTP_COUNT" -gt 0 ]; then
        pass "RTP ports in 10000+ range active ($RTP_COUNT sockets)"
    else
        warn "No RTP ports active (expected when no calls in progress)"
    fi
fi

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
TOTAL=$((PASS + FAIL + WARN))
echo -e "  ${GREEN}$PASS passed${NC}  ${RED}$FAIL failed${NC}  ${YELLOW}$WARN warnings${NC}  ($TOTAL total)"

if [ $FAIL -eq 0 ]; then
    echo -e "  ${GREEN}All critical checks passed.${NC}"
    exit 0
else
    echo -e "  ${RED}Some checks failed — review above.${NC}"
    exit 1
fi
