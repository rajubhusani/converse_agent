#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  EC2 Setup Script — Run once on a fresh Ubuntu 24.04 ARM instance
#
#  Recommended instance: t4g.large (4 vCPU ARM, 8GB) — ~$47/mo reserved
#  Region: ap-south-1 (Mumbai) for India voice traffic
#
#  Usage:
#    ssh ubuntu@<ec2-ip>
#    curl -sL https://raw.githubusercontent.com/your-repo/scripts/setup_ec2.sh | bash
#    # OR
#    scp scripts/setup_ec2.sh ubuntu@<ec2-ip>:~/
#    ssh ubuntu@<ec2-ip> 'bash setup_ec2.sh'
# ══════════════════════════════════════════════════════════════
set -euo pipefail

echo "╔══════════════════════════════════════════════════╗"
echo "║  ConverseAgent — EC2 Bootstrap                   ║"
echo "╚══════════════════════════════════════════════════╝"

# ── System Updates ───────────────────────────────────────────
echo "→ Updating system packages..."
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq

# ── Docker ───────────────────────────────────────────────────
echo "→ Installing Docker..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    echo "  Docker installed. You may need to re-login for group membership."
fi

# Docker Compose plugin (v2)
echo "→ Installing Docker Compose v2..."
sudo apt-get install -y -qq docker-compose-plugin 2>/dev/null || true

# ── Essential Tools ──────────────────────────────────────────
echo "→ Installing utilities..."
sudo apt-get install -y -qq \
    htop \
    iotop \
    net-tools \
    curl \
    jq \
    unzip \
    fail2ban \
    ufw

# ── Firewall ────────────────────────────────────────────────
echo "→ Configuring UFW firewall..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh                     # 22/tcp
sudo ufw allow 8000/tcp               # API (ALB health check)
sudo ufw allow 8765/tcp               # WebSocket
sudo ufw allow 5060/udp               # SIP signaling
sudo ufw allow 5060/tcp               # SIP signaling (TCP)
sudo ufw allow 10000:10100/udp        # RTP media
sudo ufw allow 443/tcp                # HTTPS (if running without ALB)
echo "y" | sudo ufw enable
echo "  Firewall configured."

# ── Fail2Ban ────────────────────────────────────────────────
echo "→ Configuring Fail2Ban for SSH..."
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# ── Swap (safety net for 8GB instance) ──────────────────────
echo "→ Setting up 2GB swap..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# ── Kernel Tuning for SIP/RTP ───────────────────────────────
echo "→ Tuning kernel for voice traffic..."
sudo tee /etc/sysctl.d/99-voice.conf > /dev/null <<'EOF'
# UDP buffer sizes for RTP media
net.core.rmem_max = 26214400
net.core.rmem_default = 1048576
net.core.wmem_max = 26214400
net.core.wmem_default = 1048576

# Connection tracking for SIP
net.netfilter.nf_conntrack_max = 65536

# TCP keepalive for WebSocket connections
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 6

# File descriptor limits
fs.file-max = 262144

# Disable IPv6 if not needed
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
EOF
sudo sysctl -p /etc/sysctl.d/99-voice.conf 2>/dev/null || true

# ── Docker daemon tuning ─────────────────────────────────────
echo "→ Configuring Docker daemon..."
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "50m",
        "max-file": "5"
    },
    "default-ulimits": {
        "nofile": {
            "Name": "nofile",
            "Hard": 65536,
            "Soft": 65536
        }
    },
    "live-restore": true
}
EOF
sudo systemctl restart docker

# ── Application Directory ────────────────────────────────────
echo "→ Creating application directory..."
sudo mkdir -p /opt/converse-agent
sudo chown "$USER":"$USER" /opt/converse-agent

# ── Cron: Docker cleanup ────────────────────────────────────
echo "→ Setting up Docker cleanup cron..."
(crontab -l 2>/dev/null; echo "0 3 * * * docker system prune -f --volumes 2>&1 | logger -t docker-cleanup") | crontab -

# ── Summary ─────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Setup Complete!                                 ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  Next steps:                                     ║"
echo "║  1. Re-login (for docker group):                 ║"
echo "║     exit && ssh ubuntu@<this-ip>                 ║"
echo "║                                                  ║"
echo "║  2. Clone your repo:                             ║"
echo "║     cd /opt/converse-agent                       ║"
echo "║     git clone <repo-url> .                       ║"
echo "║                                                  ║"
echo "║  3. Configure:                                   ║"
echo "║     cp .env.prod.example .env                    ║"
echo "║     nano .env  # fill in credentials             ║"
echo "║                                                  ║"
echo "║  4. Deploy:                                      ║"
echo "║     bash scripts/deploy.sh                       ║"
echo "║                                                  ║"
echo "║  Firewall ports open:                            ║"
echo "║    22 (SSH), 8000 (API), 8765 (WS),             ║"
echo "║    5060 (SIP), 10000-10100 (RTP)                ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
