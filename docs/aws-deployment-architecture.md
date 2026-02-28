# ConverseAgent — AWS Deployment Architecture

## Design Principles

The voice agent has a unique constraint that dominates every deployment decision: **sub-600ms end-to-end latency from the moment a contact stops speaking to the moment they hear the agent's reply**. This means every network hop, every container cold start, and every queue poll interval directly degrades conversation quality. The architecture below optimizes for this while keeping the non-voice channels (WhatsApp, Email, SMS, Chat) cost-effective.

The system has two fundamentally different workload profiles that should be deployed separately:

1. **Hot path** — voice calls in real-time (latency-critical, compute-heavy, stateful, WebSocket-heavy)
2. **Warm path** — orchestration, follow-up dispatch, webhook processing (throughput-oriented, stateless, queue-driven)

---

## Architecture Overview

```
                                    Route 53
                                       │
                                  ┌────▼────┐
                                  │   ALB   │
                                  │ (HTTPS) │
                                  └────┬────┘
                         ┌─────────────┼──────────────┐
                         │             │              │
                    /api/v1/*    /ws/chat/*    /webhooks/*
                         │             │              │
              ┌──────────▼──────────┐  │    ┌─────────▼─────────┐
              │   ECS Fargate       │  │    │   ECS Fargate     │
              │   "api" service     │  │    │   "worker" service│
              │   (2-8 tasks)       │  │    │   (2-6 tasks)     │
              │                     │  │    │                   │
              │  FastAPI + Uvicorn  │  │    │  FollowUpConsumer │
              │  Orchestrator       │  │    │  DelayedPromoter  │
              │  Channel Adapters   │  │    │  Rules Engine     │
              └──────────┬──────────┘  │    └─────────┬─────────┘
                         │             │              │
              ┌──────────▼─────────────▼──────────────▼─────┐
              │              ElastiCache Redis 7             │
              │         (cluster mode, 2 shards, r7g)       │
              │                                             │
              │  • Redis Streams (job queue)                 │
              │  • Pub/Sub (real-time events)                │
              │  • Session state + dedup caches              │
              └─────────────────────────────────────────────┘

              ┌─────────────────────────────────────────────┐
              │         ECS Fargate — "voice" service        │
              │         (1 task per active call)              │
              │                                              │
              │  ┌────────────────────────────────────────┐  │
              │  │         Pipecat Pipeline Process        │  │
              │  │                                        │  │
              │  │  Deepgram STT ◄──► Claude LLM          │  │
              │  │      (WS)            (WS)              │  │
              │  │                       │                │  │
              │  │                  ElevenLabs/            │  │
              │  │                  Cartesia TTS           │  │
              │  │                    (WS)                │  │
              │  │                       │                │  │
              │  │              Turn-Taking Engine         │  │
              │  │              Latency Tracker            │  │
              │  └────────────────────────────────────────┘  │
              │                      ▲                       │
              └──────────────────────│───────────────────────┘
                                     │ SIP/RTP
                              ┌──────▼──────┐
                              │   Twilio    │
                              │  SIP Trunk  │
                              │  (or Daily  │
                              │   WebRTC)   │
                              └─────────────┘
```

---

## Service-by-Service Breakdown

### 1. Networking and Ingress

**VPC** — Single VPC in `ap-south-1` (Mumbai, closest to your Bengaluru users). Three AZs with public subnets (ALB, NAT Gateway) and private subnets (ECS tasks, Redis, RDS).

**ALB (Application Load Balancer)** — Single entry point with three listener rules:

| Path Pattern | Target Group | Notes |
|---|---|---|
| `/api/v1/*`, `/webhooks/*` | api-service | Standard HTTP, health check on `/health` |
| `/ws/chat/*` | api-service | WebSocket upgrade, sticky sessions enabled |
| `/api/v1/voice/*` | voice-service | Voice callbacks routed to voice tasks |

**Why ALB and not API Gateway:** WebSocket support is native, no 29-second timeout limitation, and voice webhook callbacks need sub-50ms routing — API Gateway adds 15-30ms of overhead that matters here.

**ACM certificate** on the ALB for HTTPS. Route 53 for DNS with health-check-based failover if you go multi-region later.

**Security Groups:**
- ALB: inbound 443 from `0.0.0.0/0`
- API tasks: inbound from ALB SG only
- Redis: inbound 6379 from API + Worker + Voice SGs only
- All outbound allowed (STT/TTS/LLM provider WebSockets)

---

### 2. API Service (ECS Fargate)

**What runs here:** FastAPI application — REST endpoints, webhook receivers, WebSocket chat, Orchestrator, Channel Adapters, Rules Engine.

**Task Definition:**
```
CPU:    1024 (1 vCPU)
Memory: 2048 MB
Image:  {ECR_URI}/converse-agent-api:latest
Port:   8000
```

**Service Configuration:**
- Desired count: 2 (minimum for HA)
- Auto-scaling: 2-8 tasks, target CPU 60%, scale-in cooldown 120s
- Health check: `/health` every 15s, 2 consecutive failures to deregister
- Deployment: rolling update, minimum healthy 50%, maximum 200%

**Environment Variables** (injected from Secrets Manager + Parameter Store):
```
CONVERSE_CONFIG=/app/config/settings.yaml
REDIS_URL=redis://elasticache-cluster:6379/0
ANTHROPIC_API_KEY=arn:aws:secretsmanager:...
TWILIO_SID=arn:aws:secretsmanager:...
TWILIO_AUTH=arn:aws:secretsmanager:...
STT_API_KEY=arn:aws:secretsmanager:...
TTS_API_KEY=arn:aws:secretsmanager:...
WHATSAPP_ACCESS_TOKEN=arn:aws:secretsmanager:...
```

**Why Fargate over EC2:** The API layer is stateless (state lives in Redis), traffic is bursty (follow-up campaigns spike and subside), and you avoid managing instance patching. The 15-second Fargate cold start is acceptable here because auto-scaling keeps warm capacity.

**Why not Lambda:** WebSocket chat connections need long-lived processes. The Orchestrator holds in-memory dialogue sessions during active conversations. Uvicorn's async event loop handles concurrent webhook callbacks far more efficiently than Lambda's invocation model for this workload.

---

### 3. Worker Service (ECS Fargate)

**What runs here:** `FollowUpConsumer` (pulls jobs from Redis Streams, dispatches follow-ups), `DelayedJobPromoter` (moves scheduled jobs to active queue).

**Task Definition:**
```
CPU:    512 (0.5 vCPU)
Memory: 1024 MB
Image:  {ECR_URI}/converse-agent-worker:latest
Command: ["python", "-m", "job_queue.consumer"]
```

**Service Configuration:**
- Desired count: 2
- Auto-scaling: 2-6 tasks based on Redis Streams lag (custom CloudWatch metric)
- No ALB target group — workers pull from Redis, they don't receive HTTP traffic

**Why separate from API:** Follow-up dispatch is CPU-bound (LLM calls, template rendering) and shouldn't compete with latency-sensitive webhook processing. Scaling independently means a campaign dispatching 10,000 follow-ups doesn't slow down inbound message handling.

---

### 4. Voice Service (ECS Fargate — the critical path)

**What runs here:** One Pipecat pipeline process per active voice call. Each task holds 3 persistent WebSocket connections (STT, LLM, TTS) plus a SIP/RTP media stream.

**This is the most latency-sensitive component in the entire system.** Every architectural decision here is driven by shaving milliseconds.

**Task Definition:**
```
CPU:    2048 (2 vCPU)
Memory: 4096 MB
Image:  {ECR_URI}/converse-agent-voice:latest
Port:   8765 (Pipecat WS)
```

**Why 2 vCPU / 4 GB:** Pipecat manages concurrent async I/O across 3+ WebSocket streams plus audio processing. Under-provisioning causes event loop stalls that spike latency. The extra headroom keeps p99 latency stable.

**Service Configuration:**
- Desired count: based on expected concurrent calls (1 task handles 1-3 concurrent calls depending on CPU profile)
- Auto-scaling: scale on custom metric `ActiveVoiceCalls` pushed to CloudWatch
- Placement: spread across AZs, but prefer the AZ closest to your Twilio SIP endpoint
- **Task startup:** Pre-warm STT/TTS WebSocket connections during container init, before accepting calls. This eliminates the 200-500ms cold connection penalty on the first call.

**Networking for voice:**
- Fargate tasks need **public IP or NAT Gateway** to reach Deepgram, ElevenLabs, Cartesia, and Anthropic APIs directly
- If using Twilio SIP, Twilio sends SIP INVITE to your Pipecat endpoint — the ALB routes `/api/v1/voice/*` callbacks but the actual RTP media flows directly between Twilio's media servers and Pipecat
- If using Daily (WebRTC), Daily hosts the media server — Pipecat connects as a WebRTC peer, so you only need outbound WebSocket access

**Latency optimization — NAT Gateway placement:**
Place the NAT Gateway in the same AZ as the majority of voice tasks. Cross-AZ NAT adds 0.5-1ms per hop which compounds across 3 WebSocket streams per turn. For the voice service specifically, consider **assigning public IPs directly** to Fargate tasks (supported in `awsvpc` mode) to eliminate the NAT hop entirely.

---

### 5. ElastiCache Redis

**What it stores:**
- Redis Streams for job queue (`followup:dispatch`, `followup:delayed`)
- Voice call state (shared between API and Voice services when callbacks arrive)
- WebSocket chat session dedup and presence
- Channel adapter dedup caches (message IDs, content hashes)
- Rate limiter token buckets

**Configuration:**
```
Engine:         Redis 7.x
Node type:      cache.r7g.large (2 vCPU, 13 GB)
Cluster mode:   Disabled (single shard, 1 primary + 1 replica)
Multi-AZ:       Yes, automatic failover
Encryption:     In-transit (TLS) + at-rest (KMS)
Parameter Group: maxmemory-policy = allkeys-lru
```

**Why single shard:** The job queue and call state don't need sharding throughput — a single `r7g.large` handles 100K+ ops/sec. Redis Streams consumer groups work best on a single shard (no cross-shard coordination). Scale vertically to `r7g.xlarge` if you hit memory limits.

**Why not DynamoDB or SQS for the queue:** The system already uses Redis Streams with consumer groups for exactly-once processing. Switching to SQS would require rewriting `MessageQueue` + `Consumer` and adds 20-50ms per message vs Redis's sub-1ms. DynamoDB Streams has similar latency overhead. Redis also serves double duty as cache + pub/sub, so it's already in the infrastructure.

---

### 6. Database (Optional — RDS PostgreSQL)

The current system uses in-memory stores with Redis for persistence. For production durability of conversation history, contact records, and follow-up audit trails:

```
Engine:         PostgreSQL 16
Instance:       db.r7g.large
Multi-AZ:       Yes
Storage:        gp3, 100 GB, 3000 IOPS
Encryption:     KMS
Backup:         7-day automated, cross-region replication if needed
```

**Migration path:** Replace `ContextStore` (currently in-memory dict) with SQLAlchemy async sessions pointing to RDS. The `aiosqlite` dependency in requirements.txt already shows the abstraction is ready — swap the connection string from `sqlite+aiosqlite:///` to `postgresql+asyncpg://`.

---

### 7. Secrets and Configuration

| Secret | Service |
|---|---|
| `ANTHROPIC_API_KEY` | Secrets Manager |
| `TWILIO_SID`, `TWILIO_AUTH` | Secrets Manager |
| `DEEPGRAM_API_KEY` | Secrets Manager |
| `ELEVENLABS_API_KEY` | Secrets Manager |
| `WHATSAPP_ACCESS_TOKEN` | Secrets Manager |
| `SMTP_PASSWORD` | Secrets Manager |
| `settings.yaml` | S3 bucket, pulled at container startup |
| `dialogue_flows.yaml` | S3 bucket, hot-reloadable |
| Feature flags, thresholds | Parameter Store |

ECS task roles use IAM to access Secrets Manager — no credentials in environment variables or images.

---

### 8. Observability

**CloudWatch Metrics (custom):**
- `voice.pipeline.latency.stt_ms` — per-call STT latency
- `voice.pipeline.latency.llm_ttfb_ms` — per-call LLM time-to-first-byte
- `voice.pipeline.latency.tts_ttfb_ms` — per-call TTS time-to-first-byte
- `voice.pipeline.latency.total_ms` — end-to-end pipeline latency
- `voice.active_calls` — concurrent call gauge
- `voice.barge_in_rate` — interruption frequency (conversation quality signal)
- `voice.silence_timeouts` — calls ended due to silence
- `queue.dispatch.lag` — Redis Streams consumer lag
- `channel.{type}.failure_rate` — per-channel circuit breaker health

**CloudWatch Alarms:**
- `voice.pipeline.latency.total_ms p90 > 800ms` — PagerDuty (conversation quality degrading)
- `voice.active_calls > 80% capacity` — auto-scale voice tasks
- `channel.voice.failure_rate > 10%` — circuit breaker may be opening
- `queue.dispatch.lag > 100` — scale worker tasks

**Logs:** Structured JSON via `structlog` to CloudWatch Logs, optional export to OpenSearch for query.

**X-Ray tracing:** Instrument the Orchestrator's `handle_inbound_message` and `dispatch_followup` paths. For voice, trace each pipeline stage (STT, LLM, TTS) as separate X-Ray subsegments to pinpoint latency bottlenecks.

---

## Deployment Pipeline

```
GitHub repo
    │
    ▼
GitHub Actions / CodePipeline
    │
    ├── pytest (359 tests)
    ├── docker build
    ├── ECR push (3 images: api, worker, voice)
    │
    ▼
ECS Rolling Deployment
    ├── api-service:    rolling update, min 50% healthy
    ├── worker-service: rolling update, drain Redis Streams first
    └── voice-service:  rolling update, drain active calls first
```

**Voice deployment safety:** Before draining a voice task, wait for all active calls to complete (or timeout at `max_call_duration_s = 300`). ECS deregistration delay should be set to 310 seconds for voice tasks. New calls are routed to new tasks immediately; existing calls finish on old tasks.

**Three separate ECS services from one codebase:** Use different `CMD` overrides in each task definition. The Dockerfile stays the same:
- API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Worker: `python -m job_queue.consumer`
- Voice: `python -m voice.server` (Pipecat process manager)

---

## Cost Estimate (Moderate Load)

Assuming 50 concurrent voice calls peak, 500 follow-ups/day, 200 active chat users:

| Component | Spec | Monthly Cost (ap-south-1) |
|---|---|---|
| ECS Fargate — API (2 tasks) | 1 vCPU / 2 GB x 2 | ~$60 |
| ECS Fargate — Worker (2 tasks) | 0.5 vCPU / 1 GB x 2 | ~$25 |
| ECS Fargate — Voice (5 tasks peak) | 2 vCPU / 4 GB x 5 | ~$300 |
| ElastiCache Redis | r7g.large, Multi-AZ | ~$340 |
| RDS PostgreSQL | r7g.large, Multi-AZ | ~$380 |
| ALB | Standard, ~100 GB transfer | ~$30 |
| NAT Gateway | 2 AZs, ~50 GB data | ~$90 |
| CloudWatch | Logs + custom metrics | ~$40 |
| Secrets Manager | 10 secrets | ~$5 |
| ECR | 3 images, ~2 GB | ~$1 |
| **AWS Infrastructure Total** | | **~$1,270/mo** |

**External provider costs (usage-based, not AWS):**

| Provider | Usage Estimate | Monthly Cost |
|---|---|---|
| Anthropic Claude API | ~2M tokens/day | ~$200-600 |
| Deepgram STT | ~500 call hours/mo | ~$200 |
| ElevenLabs TTS | ~500 call hours/mo | ~$300 |
| Twilio Voice | ~500 call hours, local India | ~$500 |
| Twilio WhatsApp/SMS | ~15K messages/mo | ~$150 |
| **Provider Total** | | **~$1,350-1,750/mo** |

**Total: ~$2,600-3,000/month** at moderate scale. Voice calls dominate cost — if volume grows, Cartesia ($0.015/min vs ElevenLabs' $0.03/min) and Deepgram's volume pricing bring this down significantly.

---

## Scaling Approach

**Phase 1 (0-50 concurrent calls):** Single-region `ap-south-1`, architecture as described above. ElastiCache single shard, RDS single instance with Multi-AZ replica.

**Phase 2 (50-200 concurrent calls):** Scale voice tasks horizontally (Fargate handles up to 500 tasks per service). Upgrade ElastiCache to `r7g.xlarge`. Add CloudFront for dashboard static assets if you build the React dashboard.

**Phase 3 (200+ concurrent calls):** Add a second region (e.g., `us-east-1` for US contacts). Route 53 latency-based routing sends calls to the nearest region. Redis Global Datastore for cross-region state. Consider moving from Twilio SIP to Daily WebRTC to eliminate PSTN latency.

**Phase 3 voice-specific:** At 200+ concurrent calls, each Pipecat pipeline consumes 3 WebSocket connections to external providers. Deepgram and ElevenLabs impose concurrent connection limits per API key — you'll need to either request limit increases or pool connections across multiple API keys with the `VoiceProviderRegistry`'s named profiles.

---

## Infrastructure as Code

Recommended: **AWS CDK (Python)** since the codebase is Python. Three stacks:

1. **NetworkStack** — VPC, subnets, security groups, ALB, Route 53
2. **DataStack** — ElastiCache, RDS, S3 config bucket, Secrets Manager
3. **ComputeStack** — ECR repos, ECS cluster, 3 services (api, worker, voice), auto-scaling, CloudWatch alarms

This maps cleanly to the `docker-compose.yml` structure already in the project — the CDK stacks replace Compose for production while keeping the same container images.
