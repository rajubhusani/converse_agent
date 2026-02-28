#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  Secure SIP Ports — Restrict to telephony provider IPs
#
#  IMPORTANT: Run this AFTER deploy to lock down SIP/RTP ports.
#  By default setup_ec2.sh opens these ports to 0.0.0.0/0.
#  This script restricts them to known provider IP ranges.
#
#  Usage:
#    bash scripts/secure_sip.sh twilio     # Restrict to Twilio IPs
#    bash scripts/secure_sip.sh exotel     # Restrict to Exotel IPs
#    bash scripts/secure_sip.sh open       # Open to all (testing only)
# ══════════════════════════════════════════════════════════════
set -euo pipefail

PROVIDER="${1:-}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[secure]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
err() { echo -e "${RED}[error]${NC} $1"; exit 1; }

if [[ -z "$PROVIDER" || ! "$PROVIDER" =~ ^(twilio|exotel|plivo|open)$ ]]; then
    echo "Usage: bash scripts/secure_sip.sh <twilio|exotel|plivo|open>"
    echo ""
    echo "Restricts SIP (5060) and RTP (10000-10100) ports to provider IPs."
    echo "Use 'open' for testing only (allows all IPs)."
    exit 1
fi

# ── Provider IP Ranges ───────────────────────────────────────
# Update these from provider documentation periodically.
# Twilio: https://www.twilio.com/docs/sip-trunking/ip-addresses
# Exotel: Contact Exotel support for current IP ranges
# Plivo:  https://support.plivo.com/hc/en-us/articles/360041194072

case "$PROVIDER" in
    twilio)
        SIP_CIDRS=(
            "54.172.60.0/30"
            "54.244.51.0/30"
            "54.171.127.192/30"
            "35.156.191.128/30"
            "54.65.63.192/30"
            "54.169.127.128/30"
            "54.252.254.64/30"
            "177.71.206.192/30"
            # Twilio Edge: Mumbai
            "13.235.0.0/16"
        )
        ;;
    exotel)
        SIP_CIDRS=(
            # Exotel India (Mumbai region)
            "54.254.73.0/24"
            "13.235.0.0/16"
            "13.126.0.0/16"
            # Verify with Exotel support for current ranges
        )
        ;;
    plivo)
        SIP_CIDRS=(
            # Plivo SIP endpoints
            "54.172.0.0/16"
            "52.6.0.0/16"
            "54.252.0.0/16"
            "54.169.0.0/16"
            "13.235.0.0/16"
        )
        ;;
    open)
        warn "Opening SIP/RTP to all IPs. FOR TESTING ONLY."
        sudo ufw allow 5060/udp
        sudo ufw allow 5060/tcp
        sudo ufw allow 10000:10100/udp
        sudo ufw reload
        log "SIP/RTP ports open to all. Run with a provider name to restrict."
        exit 0
        ;;
esac

log "Securing SIP/RTP ports for: $PROVIDER"

# ── Remove existing SIP rules ────────────────────────────────
log "Removing existing SIP/RTP UFW rules..."
# Delete any existing rules for these ports
for port in "5060/udp" "5060/tcp" "10000:10100/udp"; do
    while sudo ufw status numbered | grep -q "$port"; do
        RULE_NUM=$(sudo ufw status numbered | grep "$port" | head -1 | sed 's/\[ *\([0-9]*\)\].*/\1/')
        echo "y" | sudo ufw delete "$RULE_NUM" 2>/dev/null || true
    done
done

# ── Add provider-specific rules ──────────────────────────────
for cidr in "${SIP_CIDRS[@]}"; do
    log "  Allowing $cidr → 5060/udp, 5060/tcp, 10000-10100/udp"
    sudo ufw allow from "$cidr" to any port 5060 proto udp
    sudo ufw allow from "$cidr" to any port 5060 proto tcp
    sudo ufw allow from "$cidr" to any port 10000:10100 proto udp
done

sudo ufw reload

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  SIP/RTP ports secured for: $PROVIDER"
echo "║  Allowed CIDRs: ${#SIP_CIDRS[@]} ranges"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Verify with: sudo ufw status | grep 5060       ║"
echo "╚══════════════════════════════════════════════════╝"
