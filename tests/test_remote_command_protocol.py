from __future__ import annotations

import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from os_agent.remote import (
    AdmissionOutcome,
    CommandBinding,
    CommandEnvelope,
    CommandExpiredError,
    CommandIdConflictError,
    CommandPayload,
    CommandPrincipal,
    CommandProtocolError,
    CommandStatus,
    IdentityBindingError,
    InvalidStatusTransitionError,
    NonceReplayError,
    PayloadDigestMismatchError,
    SQLiteCommandAdmissionStore,
    TimeValidationError,
    compute_payload_digest,
)

NOW = datetime(2026, 9, 6, 8, 0, tzinfo=UTC)


def envelope(
    *,
    command_id: str = "cmd-001",
    nonce: str = "nonce_0123456789abcdef",
    prompt: str = "整理今天的项目状态",
    issued_at: datetime = NOW,
    ttl: timedelta = timedelta(minutes=5),
) -> CommandEnvelope:
    return CommandEnvelope.create(
        tenant_id="tenant-1",
        user_id="user-1",
        device_id="device-1",
        command_id=command_id,
        nonce=nonce,
        payload=CommandPayload(prompt=prompt, session_id="phone-session-1"),
        issued_at=issued_at,
        ttl=ttl,
    )


PRINCIPAL = CommandPrincipal("tenant-1", "user-1", "device-1")


class CommandEnvelopeTests(unittest.TestCase):
    def test_round_trip_preserves_strongly_typed_envelope_and_digest(self) -> None:
        original = envelope()

        decoded = CommandEnvelope.from_dict(original.to_dict())

        self.assertEqual(decoded, original)
        self.assertEqual(decoded.payload.kind.value, "natural_language")
        self.assertEqual(decoded.payload_digest, compute_payload_digest(decoded.payload))
        self.assertTrue(original.to_dict()["issued_at"].endswith("Z"))

    def test_payload_digest_is_deterministic_and_tampering_is_rejected(self) -> None:
        first = CommandPayload(prompt="打开项目概览", session_id="session-1")
        second = CommandPayload(prompt="打开项目概览", session_id="session-1")
        self.assertEqual(compute_payload_digest(first), compute_payload_digest(second))

        wire = envelope().to_dict()
        wire["payload"] = dict(wire["payload"])
        wire["payload"]["prompt"] = "删除项目"
        with self.assertRaises(PayloadDigestMismatchError):
            CommandEnvelope.from_dict(wire)

    def test_payload_accepts_only_natural_language_schema(self) -> None:
        wire = envelope().to_dict()
        wire["payload"] = {
            "kind": "shell",
            "prompt": "ignored",
            "session_id": "session-1",
            "command": "rm -rf /",
        }
        with self.assertRaises(CommandProtocolError):
            CommandEnvelope.from_dict(wire)

    def test_identity_must_match_verified_tenant_user_and_device(self) -> None:
        command = envelope()
        command.assert_bound_to(PRINCIPAL)
        with self.assertRaises(IdentityBindingError):
            command.assert_bound_to(
                CommandPrincipal("tenant-1", "user-2", "device-1")
            )
        with self.assertRaises(IdentityBindingError):
            command.assert_bound_to(
                CommandPrincipal("tenant-1", "user-1", "device-2")
            )

    def test_utc_and_ttl_are_strictly_validated(self) -> None:
        with self.assertRaises(TimeValidationError):
            envelope(issued_at=NOW.replace(tzinfo=None))
        with self.assertRaises(TimeValidationError):
            envelope(
                issued_at=datetime(
                    2026, 9, 6, 16, 0, tzinfo=timezone(timedelta(hours=8))
                )
            )
        with self.assertRaises(TimeValidationError):
            envelope(ttl=timedelta(0))

        with self.assertRaises(TimeValidationError):
            envelope(ttl=timedelta(minutes=16)).validate_for_admission(now=NOW)
        with self.assertRaises(TimeValidationError):
            envelope(issued_at=NOW + timedelta(minutes=1)).validate_for_admission(
                now=NOW
            )
        with self.assertRaises(CommandExpiredError):
            envelope(issued_at=NOW - timedelta(minutes=5)).validate_for_admission(
                now=NOW
            )


class SQLiteAdmissionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.database = Path(self.temporary.name) / "remote-admission.sqlite3"
        self.store = SQLiteCommandAdmissionStore(self.database)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_admission_persists_binding_without_storing_plaintext_prompt(self) -> None:
        command = envelope(prompt="机密的自然语言指令")

        admitted = self.store.admit(command, principal=PRINCIPAL, now=NOW)
        reopened = SQLiteCommandAdmissionStore(self.database)
        record = reopened.get(command.binding)

        self.assertEqual(admitted.outcome, AdmissionOutcome.ACCEPTED)
        self.assertIsNotNone(record)
        self.assertEqual(record.payload_digest, command.payload_digest)
        self.assertEqual(record.status, CommandStatus.ADMITTED)
        with closing(sqlite3.connect(self.database)) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(remote_commands)")
            }
            serialized = " ".join(
                str(value)
                for row in connection.execute("SELECT * FROM remote_commands")
                for value in row
            )
        self.assertNotIn("payload", columns)
        self.assertNotIn("prompt", columns)
        self.assertNotIn(command.payload.prompt, serialized)

    def test_identical_command_retry_is_idempotent_across_store_instances(self) -> None:
        command = envelope()
        first = self.store.admit(command, principal=PRINCIPAL, now=NOW)
        second = SQLiteCommandAdmissionStore(self.database).admit(
            command, principal=PRINCIPAL, now=NOW
        )

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(second.outcome, AdmissionOutcome.DUPLICATE_COMMAND)
        self.assertEqual(second.record.version, 1)

    def test_command_id_collision_and_nonce_replay_are_separate_failures(self) -> None:
        self.store.admit(envelope(), principal=PRINCIPAL, now=NOW)

        with self.assertRaises(CommandIdConflictError):
            self.store.admit(
                envelope(prompt="另一条指令"), principal=PRINCIPAL, now=NOW
            )
        with self.assertRaises(NonceReplayError):
            self.store.admit(
                envelope(command_id="cmd-002"), principal=PRINCIPAL, now=NOW
            )

    def test_admission_rejects_untrusted_binding_before_writing(self) -> None:
        with self.assertRaises(IdentityBindingError):
            self.store.admit(
                envelope(),
                principal=CommandPrincipal("tenant-2", "user-1", "device-1"),
                now=NOW,
            )
        self.assertIsNone(self.store.get(envelope().binding))

    def test_get_and_transition_require_full_object_binding(self) -> None:
        command = envelope()
        self.store.admit(command, principal=PRINCIPAL, now=NOW)
        wrong_user = CommandBinding(
            "tenant-1", "user-2", "device-1", command.command_id
        )

        self.assertIsNone(self.store.get(wrong_user))
        with self.assertRaisesRegex(LookupError, "not found"):
            self.store.transition(wrong_user, CommandStatus.RUNNING, now=NOW)

    def test_state_machine_accepts_only_declared_transitions(self) -> None:
        command = envelope()
        self.store.admit(command, principal=PRINCIPAL, now=NOW)

        waiting = self.store.transition(
            command.binding, CommandStatus.WAITING_APPROVAL, now=NOW
        )
        running = self.store.transition(
            command.binding, CommandStatus.RUNNING, now=NOW
        )
        completed = self.store.transition(
            command.binding, CommandStatus.COMPLETED, now=NOW
        )

        self.assertEqual(waiting.version, 2)
        self.assertEqual(running.version, 3)
        self.assertEqual(completed.version, 4)
        with self.assertRaises(InvalidStatusTransitionError):
            self.store.transition(
                command.binding, CommandStatus.RUNNING, now=NOW
            )

    def test_state_machine_rejects_skips_and_expired_execution(self) -> None:
        command = envelope(ttl=timedelta(seconds=10))
        self.store.admit(command, principal=PRINCIPAL, now=NOW)
        with self.assertRaises(InvalidStatusTransitionError):
            self.store.transition(
                command.binding, CommandStatus.COMPLETED, now=NOW
            )
        with self.assertRaises(CommandExpiredError):
            self.store.transition(
                command.binding,
                CommandStatus.RUNNING,
                now=NOW + timedelta(seconds=10),
            )
        expired = self.store.transition(
            command.binding,
            CommandStatus.EXPIRED,
            now=NOW + timedelta(seconds=10),
        )
        self.assertEqual(expired.status, CommandStatus.EXPIRED)

    def test_concurrent_identical_delivery_is_admitted_once(self) -> None:
        command = envelope()

        def submit(_: int) -> AdmissionOutcome:
            store = SQLiteCommandAdmissionStore(self.database)
            return store.admit(command, principal=PRINCIPAL, now=NOW).outcome

        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = list(executor.map(submit, range(16)))

        self.assertEqual(outcomes.count(AdmissionOutcome.ACCEPTED), 1)
        self.assertEqual(outcomes.count(AdmissionOutcome.DUPLICATE_COMMAND), 15)


if __name__ == "__main__":
    unittest.main()
