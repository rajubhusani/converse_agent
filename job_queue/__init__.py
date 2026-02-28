"""
Message Queue — Decouples follow-up creation from dispatch.

Replaces the polling scheduler with an event-driven architecture:
- Rules engine PUBLISHES follow-up jobs to a queue
- Orchestrator CONSUMES jobs and dispatches them
- Supports Redis Streams (production) and in-memory asyncio.Queue (dev)
"""
