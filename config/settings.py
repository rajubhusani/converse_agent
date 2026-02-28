"""
Configuration loader for the ConverseAgent system.
Reads settings from YAML file with environment variable substitution.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: str = ""
    system_prompt_template: str = ""


@dataclass
class ChannelConfig:
    enabled: bool = False
    credentials: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendConfig:
    type: str = "rest"
    base_url: str = ""
    auth_type: str = "bearer"
    auth_credentials: dict[str, Any] = field(default_factory=dict)
    endpoints: dict[str, str] = field(default_factory=dict)


@dataclass
class DatabaseConfig:
    url: str = "sqlite:///./converse_agent.db"        # postgresql:// | mysql:// | sqlite://
    store_backend: str = "memory"                      # "sql" | "memory" | "file"
    store_file_dir: str = "./data"                     # directory for file backend
    redis_url: str = ""                                # used by queue, not by store


@dataclass
class QueueConfig:
    backend: str = "memory"             # "memory" for dev, "redis" for production
    redis_url: str = "redis://localhost:6379"
    consumer_group: str = "followup-workers"
    consumer_concurrency: int = 5       # max concurrent dispatches per worker
    delayed_promote_interval: int = 5   # seconds between delayed-queue scans
    retry_backoff_base: int = 60        # base seconds for exponential retry backoff


@dataclass
class Settings:
    app_name: str = "ConverseAgent"
    debug: bool = False
    secret_key: str = "change-me"
    business_domain: str = "generic"
    timezone: str = "UTC"
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    channels: dict[str, ChannelConfig] = field(default_factory=dict)
    rules: list[dict[str, Any]] = field(default_factory=list)


_settings: Optional[Settings] = None


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    pattern = re.compile(r'\$\{(\w+)\}')
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    return pattern.sub(replacer, value)


def _process_values(obj: Any) -> Any:
    """Recursively substitute env vars in all string values."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _process_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_process_values(v) for v in obj]
    return obj


def load_settings(config_path: str = None) -> Settings:
    """Load settings from YAML file."""
    global _settings

    if config_path is None:
        config_path = os.environ.get(
            "CONVERSE_CONFIG",
            str(Path(__file__).parent / "settings.yaml"),
        )

    settings = Settings()

    if Path(config_path).exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        raw = _process_values(raw)

        settings.app_name = raw.get("app_name", settings.app_name)
        settings.debug = raw.get("debug", settings.debug)
        settings.secret_key = raw.get("secret_key", settings.secret_key)
        settings.business_domain = raw.get("business_domain", settings.business_domain)
        settings.timezone = raw.get("timezone", settings.timezone)

        if "llm" in raw:
            llm = raw["llm"]
            settings.llm = LLMConfig(
                provider=llm.get("provider", "anthropic"),
                model=llm.get("model", "claude-sonnet-4-20250514"),
                temperature=llm.get("temperature", 0.7),
                max_tokens=llm.get("max_tokens", 1024),
                api_key=llm.get("api_key", ""),
                system_prompt_template=llm.get("system_prompt_template", ""),
            )

        if "database" in raw:
            db = raw["database"]
            settings.database = DatabaseConfig(
                url=db.get("url", settings.database.url),
                store_backend=db.get("store_backend", settings.database.store_backend),
                store_file_dir=db.get("store_file_dir", settings.database.store_file_dir),
                redis_url=db.get("redis_url", ""),
            )

        if "queue" in raw:
            q = raw["queue"]
            settings.queue = QueueConfig(
                backend=q.get("backend", "memory"),
                redis_url=q.get("redis_url", "redis://localhost:6379"),
                consumer_group=q.get("consumer_group", "followup-workers"),
                consumer_concurrency=q.get("consumer_concurrency", 5),
                delayed_promote_interval=q.get("delayed_promote_interval", 5),
                retry_backoff_base=q.get("retry_backoff_base", 60),
            )

        if "backend" in raw:
            be = raw["backend"]
            settings.backend = BackendConfig(
                type=be.get("type", "rest"),
                base_url=be.get("base_url", ""),
                auth_type=be.get("auth_type", "bearer"),
                auth_credentials=be.get("auth_credentials", {}),
                endpoints=be.get("endpoints", {}),
            )

        if "channels" in raw:
            for ch_name, ch_data in raw["channels"].items():
                settings.channels[ch_name] = ChannelConfig(
                    enabled=ch_data.get("enabled", False),
                    credentials=ch_data.get("credentials", {}),
                )

        settings.rules = raw.get("rules", [])

    _settings = settings
    return settings


def get_settings() -> Settings:
    """Return cached settings or load from default path."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
