"""Strongly typed protocol values for remotely submitted OSagent commands.

This module deliberately models only a natural-language request. It does not
accept a tool name, an executable, or raw shell arguments. Authentication and
signature verification happen outside this protocol; ``CommandPrincipal`` is
the trusted identity produced by that boundary and must match every envelope.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

PROTOCOL_VERSION = 1
MAX_PROMPT_CHARS = 16_000
MAX_ID_CHARS = 128
MIN_NONCE_CHARS = 16
MAX_NONCE_CHARS = 256
DEFAULT_COMMAND_TTL = timedelta(minutes=5)
DEFAULT_MAX_COMMAND_TTL = timedelta(minutes=15)
DEFAULT_CLOCK_SKEW = timedelta(seconds=30)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_NONCE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CommandProtocolError(ValueError):
    """Base class for malformed or invalid remote command data."""


class PayloadDigestMismatchError(CommandProtocolError):
    """Raised when the payload does not match its declared SHA-256 digest."""


class TimeValidationError(CommandProtocolError):
    """Raised when command timestamps or TTL are invalid."""


class CommandExpiredError(TimeValidationError):
    """Raised when a command is presented after its expiry time."""


class IdentityBindingError(PermissionError):
    """Raised when envelope identity differs from the verified caller/connection."""


class CommandKind(str, Enum):
    """Payload kinds supported by the remote command trust boundary."""

    NATURAL_LANGUAGE = "natural_language"


class CommandStatus(str, Enum):
    """Persisted command lifecycle states."""

    ADMITTED = "admitted"
    WAITING_APPROVAL = "waiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class CommandPayload:
    """The only executable input admitted by this protocol: a user prompt."""

    prompt: str
    session_id: str = ""
    kind: CommandKind = CommandKind.NATURAL_LANGUAGE

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CommandKind):
            raise CommandProtocolError("payload kind must be a CommandKind")
        if self.kind is not CommandKind.NATURAL_LANGUAGE:
            raise CommandProtocolError("unsupported command payload kind")
        if not isinstance(self.prompt, str):
            raise CommandProtocolError("prompt must be a string")
        if not self.prompt.strip():
            raise CommandProtocolError("prompt must not be empty")
        if "\x00" in self.prompt:
            raise CommandProtocolError("prompt must not contain NUL characters")
        if len(self.prompt) > MAX_PROMPT_CHARS:
            raise CommandProtocolError(
                f"prompt must not exceed {MAX_PROMPT_CHARS} characters"
            )
        if not isinstance(self.session_id, str):
            raise CommandProtocolError("session_id must be a string")
        if self.session_id:
            _validate_identifier("session_id", self.session_id)

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "prompt": self.prompt,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CommandPayload:
        _require_mapping(data, "payload")
        _reject_unknown_fields(data, {"kind", "prompt", "session_id"}, "payload")
        kind_value = _require_string(data, "kind", container="payload")
        try:
            kind = CommandKind(kind_value)
        except ValueError as exc:
            raise CommandProtocolError("unsupported command payload kind") from exc
        return cls(
            kind=kind,
            prompt=_require_string(data, "prompt", container="payload"),
            session_id=_optional_string(data, "session_id", container="payload"),
        )


def compute_payload_digest(payload: CommandPayload) -> str:
    """Return a deterministic SHA-256 digest of the canonical payload JSON."""

    if not isinstance(payload, CommandPayload):
        raise CommandProtocolError("payload must be a CommandPayload")
    canonical = json.dumps(
        payload.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class CommandPrincipal:
    """Identity established by the account token and device connection."""

    tenant_id: str
    user_id: str
    device_id: str

    def __post_init__(self) -> None:
        _validate_identifier("tenant_id", self.tenant_id)
        _validate_identifier("user_id", self.user_id)
        _validate_identifier("device_id", self.device_id)


@dataclass(frozen=True, slots=True)
class CommandBinding:
    """Object-level authorization key for one command."""

    tenant_id: str
    user_id: str
    device_id: str
    command_id: str

    def __post_init__(self) -> None:
        _validate_identifier("tenant_id", self.tenant_id)
        _validate_identifier("user_id", self.user_id)
        _validate_identifier("device_id", self.device_id)
        _validate_identifier("command_id", self.command_id)

    @property
    def principal(self) -> CommandPrincipal:
        return CommandPrincipal(self.tenant_id, self.user_id, self.device_id)


@dataclass(frozen=True, slots=True)
class CommandEnvelope:
    """Immutable command data delivered from the control plane to one device."""

    tenant_id: str
    user_id: str
    device_id: str
    command_id: str
    nonce: str
    issued_at: datetime
    expires_at: datetime
    payload: CommandPayload
    payload_digest: str
    schema_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != PROTOCOL_VERSION:
            raise CommandProtocolError(
                f"schema_version must be {PROTOCOL_VERSION}"
            )
        _validate_identifier("tenant_id", self.tenant_id)
        _validate_identifier("user_id", self.user_id)
        _validate_identifier("device_id", self.device_id)
        _validate_identifier("command_id", self.command_id)
        _validate_nonce(self.nonce)
        _require_utc("issued_at", self.issued_at)
        _require_utc("expires_at", self.expires_at)
        if self.expires_at <= self.issued_at:
            raise TimeValidationError("expires_at must be later than issued_at")
        if not isinstance(self.payload, CommandPayload):
            raise CommandProtocolError("payload must be a CommandPayload")
        if not isinstance(self.payload_digest, str) or not _SHA256_PATTERN.fullmatch(
            self.payload_digest
        ):
            raise CommandProtocolError(
                "payload_digest must be a lowercase SHA-256 hex digest"
            )
        calculated = compute_payload_digest(self.payload)
        if not hmac.compare_digest(self.payload_digest, calculated):
            raise PayloadDigestMismatchError("payload digest does not match payload")

    @property
    def binding(self) -> CommandBinding:
        return CommandBinding(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            device_id=self.device_id,
            command_id=self.command_id,
        )

    def assert_bound_to(self, principal: CommandPrincipal) -> None:
        """Require all caller and connection identity fields to match exactly."""

        if not isinstance(principal, CommandPrincipal):
            raise TypeError("principal must be a CommandPrincipal")
        supplied = (self.tenant_id, self.user_id, self.device_id)
        trusted = (principal.tenant_id, principal.user_id, principal.device_id)
        if supplied != trusted:
            raise IdentityBindingError(
                "command identity does not match the verified principal"
            )

    def validate_for_admission(
        self,
        *,
        now: datetime | None = None,
        max_ttl: timedelta = DEFAULT_MAX_COMMAND_TTL,
        clock_skew: timedelta = DEFAULT_CLOCK_SKEW,
    ) -> None:
        """Validate freshness using a timezone-aware UTC clock value."""

        current = now or datetime.now(UTC)
        _require_utc("now", current)
        if max_ttl <= timedelta(0):
            raise TimeValidationError("max_ttl must be positive")
        if clock_skew < timedelta(0):
            raise TimeValidationError("clock_skew must not be negative")
        if self.expires_at - self.issued_at > max_ttl:
            raise TimeValidationError("command TTL exceeds the configured maximum")
        if self.issued_at > current + clock_skew:
            raise TimeValidationError("issued_at is too far in the future")
        if self.expires_at <= current:
            raise CommandExpiredError("command has expired")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "command_id": self.command_id,
            "nonce": self.nonce,
            "issued_at": _utc_iso(self.issued_at),
            "expires_at": _utc_iso(self.expires_at),
            "payload": self.payload.to_dict(),
            "payload_digest": self.payload_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        tenant_id: str,
        user_id: str,
        device_id: str,
        command_id: str,
        nonce: str,
        payload: CommandPayload,
        issued_at: datetime | None = None,
        ttl: timedelta = DEFAULT_COMMAND_TTL,
    ) -> CommandEnvelope:
        if ttl <= timedelta(0):
            raise TimeValidationError("ttl must be positive")
        issued = issued_at or datetime.now(UTC)
        _require_utc("issued_at", issued)
        return cls(
            tenant_id=tenant_id,
            user_id=user_id,
            device_id=device_id,
            command_id=command_id,
            nonce=nonce,
            issued_at=issued,
            expires_at=issued + ttl,
            payload=payload,
            payload_digest=compute_payload_digest(payload),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CommandEnvelope:
        _require_mapping(data, "envelope")
        fields = {
            "schema_version",
            "tenant_id",
            "user_id",
            "device_id",
            "command_id",
            "nonce",
            "issued_at",
            "expires_at",
            "payload",
            "payload_digest",
        }
        _reject_unknown_fields(data, fields, "envelope")
        version = data.get("schema_version")
        if type(version) is not int:
            raise CommandProtocolError("envelope.schema_version must be an integer")
        payload_data = data.get("payload")
        if not isinstance(payload_data, Mapping):
            raise CommandProtocolError("envelope.payload must be an object")
        return cls(
            schema_version=version,
            tenant_id=_require_string(data, "tenant_id", container="envelope"),
            user_id=_require_string(data, "user_id", container="envelope"),
            device_id=_require_string(data, "device_id", container="envelope"),
            command_id=_require_string(data, "command_id", container="envelope"),
            nonce=_require_string(data, "nonce", container="envelope"),
            issued_at=_parse_datetime(
                _require_string(data, "issued_at", container="envelope"),
                "issued_at",
            ),
            expires_at=_parse_datetime(
                _require_string(data, "expires_at", container="envelope"),
                "expires_at",
            ),
            payload=CommandPayload.from_dict(payload_data),
            payload_digest=_require_string(
                data, "payload_digest", container="envelope"
            ),
        )


def _validate_identifier(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise CommandProtocolError(f"{name} must be a non-empty string")
    if len(value) > MAX_ID_CHARS or not _ID_PATTERN.fullmatch(value):
        raise CommandProtocolError(f"{name} contains unsupported characters")


def _validate_nonce(value: Any) -> None:
    if not isinstance(value, str):
        raise CommandProtocolError("nonce must be a string")
    if not MIN_NONCE_CHARS <= len(value) <= MAX_NONCE_CHARS:
        raise CommandProtocolError(
            f"nonce must contain {MIN_NONCE_CHARS}-{MAX_NONCE_CHARS} characters"
        )
    if not _NONCE_PATTERN.fullmatch(value):
        raise CommandProtocolError("nonce must use base64url-safe characters")


def _require_utc(name: str, value: Any) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TimeValidationError(f"{name} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise TimeValidationError(f"{name} must use UTC")


def _utc_iso(value: datetime) -> str:
    _require_utc("timestamp", value)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str, name: str) -> datetime:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise TimeValidationError(f"{name} must be an ISO-8601 datetime") from exc
    _require_utc(name, parsed)
    return parsed.astimezone(UTC)


def _require_mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise CommandProtocolError(f"{name} must be an object")


def _reject_unknown_fields(
    data: Mapping[str, Any], allowed: set[str], container: str
) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise CommandProtocolError(
            f"{container} contains unknown fields: {', '.join(unknown)}"
        )


def _require_string(
    data: Mapping[str, Any], key: str, *, container: str
) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise CommandProtocolError(f"{container}.{key} must be a string")
    return value


def _optional_string(
    data: Mapping[str, Any], key: str, *, container: str
) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise CommandProtocolError(f"{container}.{key} must be a string")
    return value
