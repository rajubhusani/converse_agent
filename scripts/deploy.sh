#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  Deploy Script — Build, validate, and launch ConverseAgent
#
#  Usage:
#    bash scripts/deploy.sh              # Full deploy
#    bash scripts/deploy.sh --build-only # Just build images
#    bash scripts/deploy.sh --restart    # Restart without rebuild
# ══════════════════════════════════════════════════════════════
set -euo pipefail

COMPOSE_FILE="docker-compose.prod.yml"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[deploy]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
err() { echo -e "${RED}[error]${NC} $1"; exit 1; }

# ── Pre-flight checks ───────────────────────────────────────
log "Running pre-flight checks..."

# Docker available?
command -v docker &>/dev/null || err "Docker not installed. Run scripts/setup_ec2.sh first."
docker compose version &>/dev/null || err "Docker Compose v2 not available."

# .env exists?
[ -f .env ] || err ".env file not found. Copy from .env.prod.example:\n  cp .env.prod.example .env"

# Required env vars set?
source .env
REQUIRED_VARS=(ANTHROPIC_API_KEY DEEPGRAM_API_KEY POSTGRES_PASSWORD SECRET_KEY)
for var in "${REQUIRED_VARS[@]}"; do
    val="${!var:-}"
    if [ -z "$val" ] || [[ "$val" == *"your_"* ]] || [[ "$val" == *"change"* ]]; then
        err "$var is not set or still has placeholder value in .env"
    fi
done

# At least one telephony provider configured?
TWILIO_OK=false
EXOTEL_OK=false
PLIVO_OK=false
[ -n "${TWILIO_SID:-}" ] && [ "${TWILIO_SID}" != "your_twilio_account_sid" ] && TWILIO_OK=true
[ -n "${EXOTEL_SID:-}" ] && [ "${EXOTEL_SID}" != "your_exotel_account_sid" ] && EXOTEL_OK=true
[ -n "${PLIVO_AUTH_ID:-}" ] && [ "${PLIVO_AUTH_ID}" != "your_plivo_auth_id" ] && PLIVO_OK=true

if ! $TWILIO_OK && ! $EXOTEL_OK && ! $PLIVO_OK; then
    warn "No telephony provider credentials configured."
    warn "Voice calls will run in stub mode (no real PSTN calls)."
    warn "Set TWILIO_SID/EXOTEL_SID/PLIVO_AUTH_ID in .env to enable."
fi

# TTS provider check
TTS_OK=false
[ -n "${CARTESIA_API_KEY:-}" ] && [ "${CARTESIA_API_KEY}" != "your_cartesia_api_key" ] && TTS_OK=true
[ -n "${ELEVENLABS_API_KEY:-}" ] && [ "${ELEVENLABS_API_KEY}" != "your_elevenlabs_api_key" ] && TTS_OK=true
$TTS_OK || warn "No TTS API key configured. Voice synthesis will not work."

log "Pre-flight checks passed."

# ── Handle flags ─────────────────────────────────────────────
case "${1:-}" in
    --restart)
        log "Restarting services (no rebuild)..."
        docker compose -f "$COMPOSE_FILE" restart
        log "Services restarted."
        exit 0
        ;;
    --build-only)
        log "Building images only..."
        docker compose -f "$COMPOSE_FILE" build
        log "Build complete."
        exit 0
        ;;
esac

# ── Build ────────────────────────────────────────────────────
log "Building Docker images..."
docker compose -f "$COMPOSE_FILE" build --parallel

# ── Database migration ───────────────────────────────────────
log "Starting database for migrations..."
docker compose -f "$COMPOSE_FILE" up -d postgres redis
sleep 5

# Wait for postgres to be ready
for i in {1..30}; do
    if docker compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U converse -d converse_agent &>/dev/null; then
        break
    fi
    sleep 1
done

log "Running database migrations..."
docker compose -f "$COMPOSE_FILE" run --rm app python scripts/migrate_db.py 2>/dev/null || warn "Migration script skipped (DB may not be fully ready)."

# ── Deploy ───────────────────────────────────────────────────
log "Deploying all services..."
docker compose -f "$COMPOSE_FILE" up -d

# ── Health check ─────────────────────────────────────────────
log "Waiting for services to be healthy..."
sleep 10

MAX_WAIT=60
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    warn "Health check timed out after ${MAX_WAIT}s. Checking logs..."
    docker compose -f "$COMPOSE_FILE" logs --tail=20 app
    err "Deployment may have issues. Check logs above."
fi

# ── Status report ────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Deployment Complete ✓                           ║"
echo "╠══════════════════════════════════════════════════╣"

# Show running containers
echo "║  Running services:                               ║"
docker compose -f "$COMPOSE_FILE" ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null | while read -r line; do
    printf "║  %-48s ║\n" "$line"
done

echo "╠══════════════════════════════════════════════════╣"
echo "║  Endpoints:                                      ║"
echo "║    API:       http://localhost:8000               ║"
echo "║    Health:    http://localhost:8000/health         ║"
echo "║    WebSocket: ws://localhost:8765                  ║"
echo "║    SIP:       sip:@${SIP_DOMAIN:-localhost}:5060   ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Webhooks (configure in telephony dashboard):    ║"
echo "║    Twilio:  ${BASE_URL:-https://your-domain}/webhooks/twilio/status"
echo "║    Exotel:  ${BASE_URL:-https://your-domain}/webhooks/exotel/status"
echo "║    Plivo:   ${BASE_URL:-https://your-domain}/webhooks/plivo/status"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Useful commands:                                ║"
echo "║    Logs:   docker compose -f $COMPOSE_FILE logs -f   "
echo "║    Stop:   docker compose -f $COMPOSE_FILE down      "
echo "║    Scale:  docker compose -f $COMPOSE_FILE up -d --scale voice=2"
echo "╚══════════════════════════════════════════════════╝"
