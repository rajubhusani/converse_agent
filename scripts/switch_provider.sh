#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  Switch Telephony Provider
#
#  Usage:
#    bash scripts/switch_provider.sh twilio     # Global (default)
#    bash scripts/switch_provider.sh exotel     # India (₹0.60/min)
#    bash scripts/switch_provider.sh plivo      # India/Global
#
#  What it does:
#  1. Validates the provider's credentials exist in .env
#  2. Updates settings.yaml with new provider + webhook URLs
#  3. Restarts voice service to pick up changes
# ══════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PROVIDER="${1:-}"
COMPOSE_FILE="docker-compose.prod.yml"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[switch]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
err() { echo -e "${RED}[error]${NC} $1"; exit 1; }

# ── Validate ─────────────────────────────────────────────────
if [[ -z "$PROVIDER" || ! "$PROVIDER" =~ ^(twilio|exotel|plivo)$ ]]; then
    echo "Usage: bash scripts/switch_provider.sh <twilio|exotel|plivo>"
    echo ""
    echo "  twilio  — Global provider (default). ~₹1.70/min for India."
    echo "  exotel  — India-only provider. ~₹0.60/min. Requires KYC + DLT."
    echo "  plivo   — India/Global. ~₹0.80/min for India."
    exit 1
fi

[ -f .env ] || err ".env file not found."
source .env

# ── Check credentials for target provider ────────────────────
case "$PROVIDER" in
    twilio)
        SID="${TWILIO_SID:-}"
        TOKEN="${TWILIO_AUTH:-}"
        PHONE="${TWILIO_PHONE:-}"
        [ -z "$SID" ] && err "TWILIO_SID not set in .env"
        [ -z "$TOKEN" ] && err "TWILIO_AUTH not set in .env"
        [ -z "$PHONE" ] && err "TWILIO_PHONE not set in .env"
        SID_VAR="\${TWILIO_SID}"
        TOKEN_VAR="\${TWILIO_AUTH}"
        PHONE_VAR="\${TWILIO_PHONE}"
        ;;
    exotel)
        SID="${EXOTEL_SID:-}"
        TOKEN="${EXOTEL_TOKEN:-}"
        PHONE="${EXOTEL_PHONE:-}"
        [ -z "$SID" ] && err "EXOTEL_SID not set in .env"
        [ -z "$TOKEN" ] && err "EXOTEL_TOKEN not set in .env"
        [ -z "$PHONE" ] && err "EXOTEL_PHONE not set in .env"
        SID_VAR="\${EXOTEL_SID}"
        TOKEN_VAR="\${EXOTEL_TOKEN}"
        PHONE_VAR="\${EXOTEL_PHONE}"
        ;;
    plivo)
        SID="${PLIVO_AUTH_ID:-}"
        TOKEN="${PLIVO_AUTH_TOKEN:-}"
        PHONE="${PLIVO_PHONE:-}"
        [ -z "$SID" ] && err "PLIVO_AUTH_ID not set in .env"
        [ -z "$TOKEN" ] && err "PLIVO_AUTH_TOKEN not set in .env"
        [ -z "$PHONE" ] && err "PLIVO_PHONE not set in .env"
        SID_VAR="\${PLIVO_AUTH_ID}"
        TOKEN_VAR="\${PLIVO_AUTH_TOKEN}"
        PHONE_VAR="\${PLIVO_PHONE}"
        ;;
esac

log "Switching telephony to: $PROVIDER"

# ── Update settings.yaml ─────────────────────────────────────
SETTINGS="config/settings.prod.yaml"
[ -f "$SETTINGS" ] || err "$SETTINGS not found."

# Use sed to swap provider-specific fields
sed -i "s|telephony_provider:.*|telephony_provider: \"$PROVIDER\"|" "$SETTINGS"
sed -i "s|telephony_account_sid:.*|telephony_account_sid: \"$SID_VAR\"|" "$SETTINGS"
sed -i "s|telephony_auth_token:.*|telephony_auth_token: \"$TOKEN_VAR\"|" "$SETTINGS"
sed -i "s|telephony_phone_number:.*|telephony_phone_number: \"$PHONE_VAR\"|" "$SETTINGS"
sed -i "s|telephony_status_callback_url:.*|telephony_status_callback_url: \"\${BASE_URL}/webhooks/${PROVIDER}/status\"|" "$SETTINGS"

log "Updated $SETTINGS"

# ── Restart voice service ────────────────────────────────────
if docker compose -f "$COMPOSE_FILE" ps --services 2>/dev/null | grep -q voice; then
    log "Restarting voice service..."
    docker compose -f "$COMPOSE_FILE" restart app voice
    sleep 5

    # Verify health
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        log "Voice service healthy after restart."
    else
        warn "Health check failed — check logs: docker compose -f $COMPOSE_FILE logs voice"
    fi
else
    log "Docker services not running. Changes will take effect on next deploy."
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Provider switched to: $PROVIDER"
printf "║  %-48s ║\n" "Phone: $PHONE"
printf "║  %-48s ║\n" "Webhook: \${BASE_URL}/webhooks/${PROVIDER}/status"
echo "╠══════════════════════════════════════════════════╣"
echo "║  REMINDER: Update your telephony dashboard:      ║"
echo "║  → Status webhook URL                            ║"
echo "║  → Answer URL                                    ║"
echo "║  → SIP endpoint trusted IPs                      ║"
echo "╚══════════════════════════════════════════════════╝"
