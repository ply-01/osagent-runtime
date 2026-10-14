"""Persistent exactly-once admission and lifecycle validation for commands."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from os_agent.remote.protocol import (
    DEFAULT_CLOCK_SKEW,
    DEFAULT_MAX_COMMAND_TTL,
    CommandBinding,
    CommandEnvelope,
    CommandExpiredError,
    CommandPrincipal,
    CommandStatus,
    TimeValidationError,
)


class AdmissionOutcome(str, Enum):
    """Whether admission created a record or matched an identical retry."""

    ACCEPTED = "accepted"
    DUPLICATE_COMMAND = "duplicate_command"


class CommandAdmissionError(RuntimeError):
    """Base class for persistent command admission failures."""


class CommandIdConflictError(CommandAdmissionError):
    """Raised when a command ID is reused with different immutable data."""


class NonceReplayError(CommandAdmissionError):
    """Raised when a device-scoped nonce is reused by another command."""


class CommandNotFoundError(LookupError):
    """Raised for an absent command or an object-level binding mismatch."""


class InvalidStatusTransitionError(CommandAdmissionError):
    """Raised when a command lifecycle transition is not permitted."""


_ALLOWED_TRANSITIONS: dict[CommandStatus, frozenset[CommandStatus]] = {
    CommandStatus.ADMITTED: frozenset(
        {
            CommandStatus.WAITING_APPROVAL,
            CommandStatus.RUNNING,
            CommandStatus.CANCELLED,
            CommandStatus.EXPIRED,
            CommandStatus.REJECTED,
        }
    ),
    CommandStatus.WAITING_APPROVAL: frozenset(
        {
            CommandStatus.RUNNING,
            CommandStatus.CANCELLED,
            CommandStatus.EXPIRED,
            CommandStatus.REJECTED,
        }
    ),
    CommandStatus.RUNNING: frozenset(
        {CommandStatus.COMPLETED, CommandStatus.FAILED}
    ),
    CommandStatus.COMPLETED: frozenset(),
    CommandStatus.FAILED: frozenset(),
    CommandStatus.CANCELLED: frozenset(),
    CommandStatus.EXPIRED: frozenset(),
    CommandStatus.REJECTED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class CommandRecord:
    """Immutable view of the metadata retained by the admission store."""

    binding: CommandBinding
    nonce: str
    payload_digest: str
    issued_at: datetime
    expires_at: datetime
    status: CommandStatus
    admitted_at: datetime
    updated_at: datetime
    version: int


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    outcome: AdmissionOutcome
    record: CommandRecord

    @property
    def accepted(self) -> bool:
        return self.outcome is AdmissionOutcome.ACCEPTED


def allowed_transitions(status: CommandStatus) -> frozenset[CommandStatus]:
    """Return the explicit set of next states for a lifecycle state."""

    if not isinstance(status, CommandStatus):
        raise TypeError("status must be a CommandStatus")
    return _ALLOWED_TRANSITIONS[status]


class SQLiteCommandAdmissionStore:
    """Durably admit commands and reject ID/nonce replay across restarts.

    The store intentionally retains no prompt text. The cloud broker or caller
    may hold the encrypted payload while this local database keeps only the
    immutable identity binding and digest needed for safe de-duplication.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        max_ttl: timedelta = DEFAULT_MAX_COMMAND_TTL,
        clock_skew: timedelta = DEFAULT_CLOCK_SKEW,
        timeout_seconds: float = 5.0,
    ) -> None:
        if max_ttl <= timedelta(0):
            raise ValueError("max_ttl must be positive")
        if clock_skew < timedelta(0):
            raise ValueError("clock_skew must not be negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.path = Path(path)
        self.max_ttl = max_ttl
        self.clock_skew = clock_skew
        self.timeout_seconds = timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def admit(
        self,
        envelope: CommandEnvelope,
        *,
        principal: CommandPrincipal,
        now: datetime | None = None,
    ) -> AdmissionResult:
        """Atomically admit a fresh command or return its identical prior record."""

        if not isinstance(envelope, CommandEnvelope):
            raise TypeError("envelope must be a CommandEnvelope")
        envelope.assert_bound_to(principal)
        current = now or datetime.now(UTC)
        envelope.validate_for_admission(
            now=current,
            max_ttl=self.max_ttl,
            clock_skew=self.clock_skew,
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM remote_commands WHERE command_id = ?",
                (envelope.command_id,),
            ).fetchone()
            if existing is not None:
                record = self._record(existing)
                if self._same_envelope(record, envelope):
                    return AdmissionResult(
                        outcome=AdmissionOutcome.DUPLICATE_COMMAND,
                        record=record,
                    )
                raise CommandIdConflictError(
                    "command_id is already bound to different immutable command data"
                )

            replay = connection.execute(
                """
                SELECT command_id FROM remote_commands
                WHERE tenant_id = ? AND device_id = ? AND nonce = ?
                """,
                (envelope.tenant_id, envelope.device_id, envelope.nonce),
            ).fetchone()
            if replay is not None:
                raise NonceReplayError(
                    "nonce was already admitted for this tenant and device"
                )

            timestamp = _utc_iso(current)
            connection.execute(
                """
                INSERT INTO remote_commands (
                    command_id, tenant_id, user_id, device_id, nonce,
                    payload_digest, issued_at, expires_at, status,
                    admitted_at, updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    envelope.command_id,
                    envelope.tenant_id,
                    envelope.user_id,
                    envelope.device_id,
                    envelope.nonce,
                    envelope.payload_digest,
                    _utc_iso(envelope.issued_at),
                    _utc_iso(envelope.expires_at),
                    CommandStatus.ADMITTED.value,
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM remote_commands WHERE command_id = ?",
                (envelope.command_id,),
            ).fetchone()
            assert row is not None
            return AdmissionResult(
                outcome=AdmissionOutcome.ACCEPTED,
                record=self._record(row),
            )

    def get(self, binding: CommandBinding) -> CommandRecord | None:
        """Read a command only when every object-level binding field matches."""

        if not isinstance(binding, CommandBinding):
            raise TypeError("binding must be a CommandBinding")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM remote_commands
                WHERE command_id = ? AND tenant_id = ? AND user_id = ? AND device_id = ?
                """,
                (
                    binding.command_id,
                    binding.tenant_id,
                    binding.user_id,
                    binding.device_id,
                ),
            ).fetchone()
        return None if row is None else self._record(row)

    def transition(
        self,
        binding: CommandBinding,
        target: CommandStatus,
        *,
        now: datetime | None = None,
    ) -> CommandRecord:
        """Apply one allowed state transition under the full command binding."""

        if not isinstance(binding, CommandBinding):
            raise TypeError("binding must be a CommandBinding")
        if not isinstance(target, CommandStatus):
            raise TypeError("target must be a CommandStatus")
        current_time = now or datetime.now(UTC)
        _require_utc_now(current_time)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM remote_commands
                WHERE command_id = ? AND tenant_id = ? AND user_id = ? AND device_id = ?
                """,
                (
                    binding.command_id,
                    binding.tenant_id,
                    binding.user_id,
                    binding.device_id,
                ),
            ).fetchone()
            if row is None:
                raise CommandNotFoundError("command not found for the supplied binding")
            record = self._record(row)
            permitted = allowed_transitions(record.status)
            if target not in permitted:
                raise InvalidStatusTransitionError(
                    f"cannot transition command from {record.status.value} to {target.value}"
                )
            if (
                record.status
                in {CommandStatus.ADMITTED, CommandStatus.WAITING_APPROVAL}
                and current_time >= record.expires_at
                and target is not CommandStatus.EXPIRED
            ):
                raise CommandExpiredError(
                    "expired command must transition to expired before execution"
                )
            connection.execute(
                """
                UPDATE remote_commands
                SET status = ?, updated_at = ?, version = version + 1
                WHERE command_id = ? AND tenant_id = ? AND user_id = ? AND device_id = ?
                """,
                (
                    target.value,
                    _utc_iso(current_time),
                    binding.command_id,
                    binding.tenant_id,
                    binding.user_id,
                    binding.device_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM remote_commands WHERE command_id = ?",
                (binding.command_id,),
            ).fetchone()
            assert updated is not None
            return self._record(updated)

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS remote_commands (
                    command_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    admitted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 1),
                    UNIQUE (tenant_id, device_id, nonce)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS remote_commands_binding_idx
                ON remote_commands (tenant_id, user_id, device_id, command_id)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            if connection.in_transaction:
                connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _same_envelope(record: CommandRecord, envelope: CommandEnvelope) -> bool:
        return (
            record.binding == envelope.binding
            and record.nonce == envelope.nonce
            and record.payload_digest == envelope.payload_digest
            and record.issued_at == envelope.issued_at
            and record.expires_at == envelope.expires_at
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> CommandRecord:
        try:
            status = CommandStatus(str(row["status"]))
        except ValueError as exc:
            raise RuntimeError("admission store contains an unknown command status") from exc
        return CommandRecord(
            binding=CommandBinding(
                tenant_id=str(row["tenant_id"]),
                user_id=str(row["user_id"]),
                device_id=str(row["device_id"]),
                command_id=str(row["command_id"]),
            ),
            nonce=str(row["nonce"]),
            payload_digest=str(row["payload_digest"]),
            issued_at=_parse_utc(str(row["issued_at"])),
            expires_at=_parse_utc(str(row["expires_at"])),
            status=status,
            admitted_at=_parse_utc(str(row["admitted_at"])),
            updated_at=_parse_utc(str(row["updated_at"])),
            version=int(row["version"]),
        )


def _utc_iso(value: datetime) -> str:
    _require_utc_now(value)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise RuntimeError("admission store contains an invalid timestamp") from exc
    _require_utc_now(parsed)
    return parsed.astimezone(UTC)


def _require_utc_now(value: Any) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TimeValidationError("now must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise TimeValidationError("now must use UTC")
