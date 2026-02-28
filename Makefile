# ══════════════════════════════════════════════════════════════
#  ConverseAgent — Makefile
#
#  Usage:
#    make dev        — Start development environment
#    make prod       — Deploy production stack
#    make test       — Run all tests
#    make smoke      — Run smoke tests against running instance
#    make switch-exotel — Switch to Exotel provider
#    make logs       — Tail all service logs
#    make costs      — Show cost projections
# ══════════════════════════════════════════════════════════════

.PHONY: help dev prod test smoke logs costs clean migrate status

# Default
help:
	@echo "ConverseAgent Commands:"
	@echo ""
	@echo "  Development:"
	@echo "    make dev            Start dev environment (docker-compose.yml)"
	@echo "    make dev-down       Stop dev environment"
	@echo "    make test           Run all tests"
	@echo "    make test-voice     Run voice/telephony tests only"
	@echo "    make test-backends  Run database/store/queue backend tests"
	@echo ""
	@echo "  Production (PostgreSQL + Redis):"
	@echo "    make prod           Deploy production stack"
	@echo "    make prod-restart   Restart without rebuild"
	@echo "    make prod-down      Stop production stack"
	@echo "    make migrate        Run database migrations"
	@echo ""
	@echo "  MySQL Stack:"
	@echo "    make mysql-up       Start MySQL-backed stack"
	@echo "    make mysql-down     Stop MySQL stack"
	@echo "    make mysql-migrate  Run MySQL migrations"
	@echo ""
	@echo "  Lightweight (No DB/Redis):"
	@echo "    make lite-up        Start in-memory/file stack"
	@echo "    make lite-down      Stop lite stack"
	@echo ""
	@echo "  Provider Switching:"
	@echo "    make switch-twilio  Switch to Twilio (default)"
	@echo "    make switch-exotel  Switch to Exotel (India savings)"
	@echo "    make switch-plivo   Switch to Plivo"
	@echo ""
	@echo "  Monitoring:"
	@echo "    make costs          Show cost projections"
	@echo "    make rates          Show rate card comparison"
	@echo "    make smoke          Run smoke tests"
	@echo "    make status         Show service status"
	@echo "    make logs           Tail all logs"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean          Remove containers, images, volumes"
	@echo "    make setup-ec2      Run EC2 setup script"

# ── Development ──────────────────────────────────────────────
dev:
	docker compose up -d
	@echo "Dev environment running at http://localhost:8000"

dev-down:
	docker compose down

test:
	python -m pytest tests/ --ignore=tests/test_integration.py -v

test-voice:
	python -m pytest tests/test_telephony_factory.py tests/test_voice_subsystem.py -v

test-backends:
	python -m pytest tests/test_backend_switching.py -v

test-all:
	python -m pytest tests/ -v

# ── Production ───────────────────────────────────────────────
PROD_COMPOSE = docker-compose.prod.yml

prod:
	bash scripts/deploy.sh

prod-build:
	bash scripts/deploy.sh --build-only

prod-restart:
	bash scripts/deploy.sh --restart

prod-down:
	docker compose -f $(PROD_COMPOSE) down

# ── MySQL Production ─────────────────────────────────────────
mysql-up:
	docker compose -f docker-compose.mysql.yml up -d
	@echo "MySQL stack running. API: http://localhost:8000"

mysql-down:
	docker compose -f docker-compose.mysql.yml down

mysql-migrate:
	docker compose -f docker-compose.mysql.yml run --rm app python scripts/migrate_db.py

# ── Lightweight (No Database, No Redis) ──────────────────────
lite-up:
	docker compose -f docker-compose.lite.yml up -d
	@echo "Lite stack running (in-memory). API: http://localhost:8000"

lite-down:
	docker compose -f docker-compose.lite.yml down

migrate:
	docker compose -f $(PROD_COMPOSE) run --rm app python scripts/migrate_db.py

migrate-check:
	docker compose -f $(PROD_COMPOSE) run --rm app python scripts/migrate_db.py --check

smoke:
	bash scripts/smoke_test.sh

status:
	@docker compose -f $(PROD_COMPOSE) ps

logs:
	docker compose -f $(PROD_COMPOSE) logs -f --tail=50

logs-voice:
	docker compose -f $(PROD_COMPOSE) logs -f --tail=100 voice

logs-app:
	docker compose -f $(PROD_COMPOSE) logs -f --tail=100 app

# ── Provider Switching ───────────────────────────────────────
switch-twilio:
	bash scripts/switch_provider.sh twilio

switch-exotel:
	bash scripts/switch_provider.sh exotel

switch-plivo:
	bash scripts/switch_provider.sh plivo

# ── Monitoring ───────────────────────────────────────────────
costs:
	python scripts/cost_monitor.py --estimate

rates:
	python scripts/cost_monitor.py --rates

# ── Maintenance ──────────────────────────────────────────────
clean:
	docker compose -f $(PROD_COMPOSE) down -v --rmi local
	docker compose down -v --rmi local
	docker system prune -f

setup-ec2:
	bash scripts/setup_ec2.sh

# ── Nginx ────────────────────────────────────────────────────
nginx-install:
	sudo cp nginx/converse-agent.conf /etc/nginx/sites-available/
	sudo ln -sf /etc/nginx/sites-available/converse-agent.conf /etc/nginx/sites-enabled/
	sudo nginx -t
	sudo systemctl reload nginx
	@echo "Nginx configured. Run: sudo certbot --nginx -d voice.yourdomain.com"
