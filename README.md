# ConverseAgent

Multi-channel follow-up automation system with configurable business state machines. Automates business conversations across Chat, Email, WhatsApp, and Voice (Pipecat) with cross-channel continuity, LLM-powered response generation, and declarative workflow configuration.

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │                  FastAPI Backend                 │
                    │                                                  │
  Inbound ──────── │  Orchestrator ◄──► State Machine (YAML configs)  │
  (webhooks/WS)    │       │                     │                    │
                    │       ▼                     ▼                    │
                    │  Context Tracker      TransitionActions          │
                    │  (cross-channel)      (enqueue/resolve/          │
                    │       │                escalate/update)          │
                    │       ▼                                          │
                    │  Conversation Engine ──► Claude LLM              │
                    │       │                                          │
                    │       ▼                                          │
  Outbound ◄────── │  Channel Adapters                                │
  (WA/Email/       │  (Chat/Email/WA/Voice)                           │
   Voice/Chat)     │                                                  │
                    │  Rules Engine ──► Message Queue ──► Consumer     │
                    │                  (Redis Streams)                 │
                    └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                                   Backend System
                                   (REST/GraphQL)
```

## Key Concepts

### Business State Machines
All business logic is configuration-driven via YAML state maps. No hardcoded intent→action branching. Each business process (payment collection, order fulfillment, etc.) is modeled as a finite state machine.

```yaml
state_maps:
  - process_type: payment_collection
    states: [pending, reminded, acknowledged, promised, confirmed, overdue, escalated, closed]
    initial_state: pending
    terminal_states: [confirmed, closed]
    transitions:
      - from_states: [reminded]
        to_state: acknowledged
        trigger: { type: intent, value: acknowledged }
        actions:
          - { type: schedule_check, delay_minutes: 2880 }
```

### Context Changes Drive Everything
Every event in the system is a **context change** that flows through the state machine:
- **Intents**: LLM-detected from inbound messages (e.g. `payment_confirmed`, `complaint`)
- **Events**: Backend system events (e.g. `payment_received`, `order_shipped`)
- **Timeouts**: Scheduled checks that fire when no response arrives
- **Actions**: System actions (e.g. `outreach_sent`)
- **Manual**: Operator overrides via API

### Multi-Process Binding
A single conversation can track multiple business processes simultaneously. For example, an order conversation can have both `payment_collection` and `order_fulfillment` state bindings running in parallel.

### Queue-Driven Dispatch
Follow-ups are published to a Redis Streams message queue and consumed by worker tasks. Delayed jobs are promoted to the dispatch queue when their `scheduled_at` time arrives. Failed jobs retry with exponential backoff, then move to a dead-letter queue.

## Quick Start

### Prerequisites
- Python 3.11+
- Redis (for production queue; in-memory queue available for development)
- Anthropic API key (for LLM features)

### Setup

```bash
# Clone and install
pip install -r requirements.txt

# Configure
cp config/settings.example.yaml config/settings.yaml
cp .env.example .env
# Edit .env with your API keys

# Run
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker compose up -d
```

## Configuration

All configuration lives in `config/settings.yaml`:

| Section | Purpose |
|---------|---------|
| `llm` | Claude model, temperature, API key |
| `channels` | Enable/configure Chat, Email, WhatsApp, Voice |
| `backend` | Business system REST API connection |
| `queue` | Redis Streams or in-memory queue settings |
| `rules` | Event/schedule-triggered follow-up rules |
| `state_maps` | Business process state machine definitions |

Environment variables can be referenced as `${VAR_NAME}` in the YAML.

## API Endpoints

### Core
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/messages/inbound` | Receive inbound message (any channel) |
| POST | `/api/v1/events` | Receive backend system events |
| POST | `/api/v1/context-change` | Manually apply a context change |
| POST | `/api/v1/process` | Evaluate rules against backend data |

### State Machines
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/state-maps` | List registered state machine definitions |
| GET | `/api/v1/state-maps/{process_type}` | Get specific state map |
| GET | `/api/v1/state-bindings` | List bindings (filterable) |
| GET | `/api/v1/state-bindings/{binding_id}` | Get binding with full history |
| POST | `/api/v1/state-bindings` | Manually bind a process to a conversation |
| GET | `/api/v1/conversations/{id}/states` | Get all process states for a conversation |

### Management
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/stats` | Dashboard statistics |
| GET | `/api/v1/conversations` | List conversations |
| GET | `/api/v1/followups` | List follow-ups |
| GET | `/api/v1/contacts` | List contacts |
| GET | `/api/v1/queue/stats` | Queue depths |
| POST | `/api/v1/queue/dlq/replay` | Replay dead-letter jobs |

### Webhooks
| Method | Path | Description |
|--------|------|-------------|
| POST | `/webhooks/whatsapp` | WhatsApp Cloud API webhook |
| POST | `/webhooks/email` | Email inbound parse webhook |
| POST | `/api/v1/voice/transcript` | Pipecat voice transcript |
| WS | `/ws/chat/{contact_id}` | Real-time chat WebSocket |

## Project Structure

```
converse-agent/
├── api/main.py                  # FastAPI app, routes, webhooks, WebSocket
├── core/
│   ├── orchestrator.py          # Central coordinator + action executor
│   └── engine.py                # LLM-powered response generation + intent detection
├── context/
│   ├── tracker.py               # Cross-channel context, state binding management
│   └── state_machine.py         # Business process FSM engine
├── models/schemas.py            # All data models (Pydantic)
├── rules/engine.py              # Rule evaluation + follow-up creation
├── channels/
│   ├── base.py                  # Channel adapter interface + registry
│   ├── chat_adapter.py          # WebSocket chat
│   ├── email_adapter.py         # SMTP/SendGrid email
│   ├── whatsapp_adapter.py      # WhatsApp Cloud API
│   └── voice_adapter.py         # Pipecat voice
├── backend/connector.py         # Business system integration (REST + mock)
├── queue/
│   ├── message_queue.py         # Redis Streams + in-memory queue
│   └── consumer.py              # Job consumer + delayed promoter
├── utils/conditions.py          # Shared condition evaluator
├── config/
│   ├── settings.py              # Config loader with env var substitution
│   └── settings.example.yaml    # Full configuration reference
├── dashboard.jsx                # React admin dashboard
├── tests/                       # Test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Extending

### Add a New Business Process
Add a new state map to `config/settings.yaml` under `state_maps:`. No code changes needed.

### Add a New Channel
1. Create a new adapter extending `ChannelAdapter` in `channels/`
2. Register it in `api/main.py`

### Add a Custom Backend
Subclass `BackendConnector` in `backend/connector.py` and implement the abstract methods.

### Persistent Storage
The default `ContextStore` is in-memory. For production, implement database-backed storage by subclassing `ContextStore` with SQLAlchemy/Redis persistence.

## License

Internal use.
