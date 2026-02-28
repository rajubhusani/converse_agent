"""
Backend Connector — Generic adapter for any business backend system.

Supports REST, GraphQL, gRPC, and direct database connections.
The connector is configured via settings.yaml and provides a uniform
interface for the rest of the system to fetch business data.
"""
from __future__ import annotations

import abc
import json
import structlog
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import BackendConfig, get_settings
from models.schemas import Contact, ContactChannel, ChannelType

logger = structlog.get_logger()


class BackendConnector(abc.ABC):
    """Abstract base for all backend connectors."""

    @abc.abstractmethod
    async def get_pending_followups(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]:
        """Fetch follow-ups that need action from the backend."""
        ...

    @abc.abstractmethod
    async def get_contact(self, contact_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single contact by ID."""
        ...

    @abc.abstractmethod
    async def get_business_context(self, followup_id: str) -> dict[str, Any]:
        """Fetch business context for a specific follow-up."""
        ...

    @abc.abstractmethod
    async def update_followup_status(self, followup_id: str, status: str, outcome: str = "") -> bool:
        """Report follow-up outcome back to the backend."""
        ...

    @abc.abstractmethod
    async def search_contacts(self, query: str) -> list[dict[str, Any]]:
        """Search contacts in the backend."""
        ...

    @abc.abstractmethod
    async def call_endpoint(self, endpoint: str, payload: dict[str, Any] = None) -> dict[str, Any]:
        """
        Call an arbitrary backend endpoint with a payload.
        Used by the state machine action executor for update_backend actions.
        The endpoint can be a named key from settings or a raw URL path.
        """
        ...

    def normalize_contact(self, raw: dict[str, Any]) -> Contact:
        """
        Convert raw backend data to a Contact model.
        Override this in domain-specific subclasses for custom mapping.
        """
        channels = []
        # Auto-detect channel addresses from common field names
        channel_fields = {
            ChannelType.EMAIL: ["email", "email_address", "mail"],
            ChannelType.WHATSAPP: ["whatsapp", "whatsapp_number", "phone", "mobile", "phone_number"],
            ChannelType.CHAT: ["chat_id", "user_id"],
            ChannelType.VOICE: ["phone", "mobile", "phone_number", "landline"],
        }
        for ch_type, fields in channel_fields.items():
            for field in fields:
                if field in raw and raw[field]:
                    channels.append(ContactChannel(
                        channel=ch_type,
                        address=str(raw[field]),
                        preferred=(field in ("whatsapp", "email")),
                    ))
                    break  # one per channel type

        return Contact(
            external_id=str(raw.get("id", raw.get("external_id", ""))),
            name=raw.get("name", raw.get("full_name", raw.get("contact_name", "Unknown"))),
            role=raw.get("role", raw.get("type", raw.get("contact_type", ""))),
            organization=raw.get("organization", raw.get("company", raw.get("org", ""))),
            channels=channels,
            metadata=raw,
        )


class RESTBackendConnector(BackendConnector):
    """
    REST API backend connector.
    Calls configured endpoints to fetch/update business data.
    """

    def __init__(self, config: BackendConfig = None):
        self.config = config or get_settings().backend
        self.client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            headers = {}
            if self.config.auth_type == "bearer":
                token = self.config.auth_credentials.get("token", "")
                headers["Authorization"] = f"Bearer {token}"
            elif self.config.auth_type == "api_key":
                key_name = self.config.auth_credentials.get("header_name", "X-API-Key")
                headers[key_name] = self.config.auth_credentials.get("api_key", "")

            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=30.0,
            )
        return self.client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    async def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        client = await self._get_client()
        url = self.config.endpoints.get(endpoint, endpoint)
        # Replace path parameters
        for k, v in kwargs.pop("path_params", {}).items():
            url = url.replace(f"{{{k}}}", str(v))

        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    async def get_pending_followups(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]:
        try:
            result = await self._request("GET", "get_followups", params=filters or {})
            return result if isinstance(result, list) else result.get("data", result.get("results", []))
        except Exception as e:
            logger.error("backend_fetch_followups_failed", error=str(e))
            return []

    async def get_contact(self, contact_id: str) -> Optional[dict[str, Any]]:
        try:
            return await self._request(
                "GET", "get_contact",
                path_params={"contact_id": contact_id},
            )
        except Exception as e:
            logger.error("backend_fetch_contact_failed", contact_id=contact_id, error=str(e))
            return None

    async def get_business_context(self, followup_id: str) -> dict[str, Any]:
        try:
            return await self._request(
                "GET", "get_context",
                path_params={"followup_id": followup_id},
            )
        except Exception as e:
            logger.error("backend_fetch_context_failed", followup_id=followup_id, error=str(e))
            return {}

    async def update_followup_status(self, followup_id: str, status: str, outcome: str = "") -> bool:
        try:
            await self._request(
                "PUT", "update_followup",
                path_params={"followup_id": followup_id},
                json={"status": status, "outcome": outcome},
            )
            return True
        except Exception as e:
            logger.error("backend_update_failed", followup_id=followup_id, error=str(e))
            return False

    async def search_contacts(self, query: str) -> list[dict[str, Any]]:
        try:
            result = await self._request("GET", "search_contacts", params={"q": query})
            return result if isinstance(result, list) else result.get("data", result.get("results", []))
        except Exception as e:
            logger.error("backend_search_failed", error=str(e))
            return []

    async def call_endpoint(self, endpoint: str, payload: dict[str, Any] = None) -> dict[str, Any]:
        try:
            return await self._request("POST", endpoint, json=payload or {})
        except Exception as e:
            logger.error("backend_call_endpoint_failed", endpoint=endpoint, error=str(e))
            return {"status": "error", "error": str(e)}

    async def close(self):
        if self.client:
            await self.client.aclose()


class MockBackendConnector(BackendConnector):
    """
    Mock backend for development and testing.
    Returns sample data mimicking a Dealer Management System.
    """

    def __init__(self):
        self._contacts = {
            "D001": {
                "id": "D001", "name": "Rajesh Kumar", "role": "dealer",
                "organization": "Kumar Motors", "email": "rajesh@kumarmotors.com",
                "whatsapp": "+919876543210", "phone": "+919876543210",
                "region": "North", "tier": "gold",
            },
            "D002": {
                "id": "D002", "name": "Priya Sharma", "role": "dealer",
                "organization": "Sharma Auto", "email": "priya@sharmaauto.in",
                "whatsapp": "+919876543211", "phone": "+919876543211",
                "region": "West", "tier": "platinum",
            },
            "V001": {
                "id": "V001", "name": "AutoParts Global", "role": "vendor",
                "organization": "AutoParts Global Ltd", "email": "orders@autopartsglobal.com",
                "phone": "+911234567890",
                "region": "International", "tier": "strategic",
            },
        }
        self._followups = [
            {
                "id": "FU001", "contact_id": "D001", "type": "payment_reminder",
                "payment_status": "overdue", "days_overdue": 7,
                "amount": 250000, "currency": "INR", "invoice_number": "INV-2024-1234",
                "due_date": "2024-12-01",
            },
            {
                "id": "FU002", "contact_id": "D002", "type": "order_confirmation",
                "order_id": "ORD-2024-5678", "order_value": 500000,
                "items": ["Brake Pads x100", "Oil Filters x200"],
                "expected_delivery": "2024-12-20",
            },
            {
                "id": "FU003", "contact_id": "V001", "type": "delivery_tracking",
                "po_number": "PO-2024-9012", "shipment_id": "SHIP-3456",
                "status": "in_transit", "eta": "2024-12-15",
            },
        ]

    async def get_pending_followups(self, filters=None) -> list[dict[str, Any]]:
        return self._followups

    async def get_contact(self, contact_id: str) -> Optional[dict[str, Any]]:
        return self._contacts.get(contact_id)

    async def get_business_context(self, followup_id: str) -> dict[str, Any]:
        for fu in self._followups:
            if fu["id"] == followup_id:
                contact = self._contacts.get(fu["contact_id"], {})
                return {**fu, "contact": contact}
        return {}

    async def update_followup_status(self, followup_id: str, status: str, outcome: str = "") -> bool:
        logger.info("mock_backend_update", followup_id=followup_id, status=status, outcome=outcome)
        return True

    async def search_contacts(self, query: str) -> list[dict[str, Any]]:
        return [c for c in self._contacts.values() if query.lower() in json.dumps(c).lower()]

    async def call_endpoint(self, endpoint: str, payload: dict[str, Any] = None) -> dict[str, Any]:
        logger.info("mock_backend_call_endpoint", endpoint=endpoint, payload_keys=list((payload or {}).keys()))
        return {"status": "ok", "endpoint": endpoint, "mock": True}


def create_backend_connector(config: BackendConfig = None) -> BackendConnector:
    """Factory function to create the appropriate backend connector."""
    config = config or get_settings().backend
    if config.type == "rest" and config.base_url:
        return RESTBackendConnector(config)
    # Add GraphQL, gRPC, Database connectors here
    logger.warning("using_mock_backend", reason="no backend configured or base_url empty")
    return MockBackendConnector()
