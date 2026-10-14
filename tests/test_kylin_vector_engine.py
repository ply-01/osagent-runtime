from __future__ import annotations

import ctypes
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from os_agent.memory.kylin_vector_engine import (
    KylinVectorEngine,
    _bridge_abi_error,
    _stage_library,
    kylin_vector_bridge_available,
    kylin_vector_bridge_load_error,
    kylin_vector_bridge_path,
    kylin_vector_bridge_status,
)


class KylinVectorBridgeArtifactTests(unittest.TestCase):
    def test_bridge_discovery_does_not_load_library(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "libbridge.so"
            source.write_bytes(b"bridge")
            self.assertEqual(kylin_vector_bridge_path(str(source)), source)
            self.assertTrue(kylin_vector_bridge_available(str(source)))

    def test_bridge_is_staged_with_source_signature(self) -> None:
        with TemporaryDirectory() as source_dir, TemporaryDirectory() as stage_dir:
            source = Path(source_dir) / "libbridge.so"
            source.write_bytes(b"bridge")
            old = os.environ.get("OS_AGENT_KYLIN_VECTOR_STAGE_DIR")
            os.environ["OS_AGENT_KYLIN_VECTOR_STAGE_DIR"] = stage_dir
            try:
                staged = Path(_stage_library(source))
            finally:
                if old is None:
                    os.environ.pop("OS_AGENT_KYLIN_VECTOR_STAGE_DIR", None)
                else:
                    os.environ["OS_AGENT_KYLIN_VECTOR_STAGE_DIR"] = old
            self.assertNotEqual(staged, source)
            self.assertEqual(staged.read_bytes(), b"bridge")
            self.assertTrue(staged.name.startswith("osagent-kylin-vector-"))

    def test_bridge_uses_installed_path_without_explicit_staging(self) -> None:
        with TemporaryDirectory() as tmp:
            source = (Path(tmp) / "libbridge.so").resolve()
            source.write_bytes(b"bridge")
            with mock.patch.dict(
                os.environ,
                {"OS_AGENT_KYLIN_VECTOR_STAGE_DIR": ""},
            ):
                resolved = Path(_stage_library(source))

            self.assertEqual(resolved, source)

    def test_bridge_load_error_reports_an_incompatible_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "libbridge.so"
            source.write_bytes(b"bridge")
            with mock.patch(
                "os_agent.memory.kylin_vector_engine._stage_library",
                return_value=str(source),
            ), mock.patch(
                "os_agent.memory.kylin_vector_engine.ctypes.CDLL",
                side_effect=OSError("undefined symbol: VectorDB::IndexDesc"),
            ):
                error = kylin_vector_bridge_load_error(str(source))

        self.assertEqual(
            error,
            "OSError: undefined symbol: VectorDB::IndexDesc",
        )

    def test_bridge_status_distinguishes_missing_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            status = kylin_vector_bridge_status(str(Path(tmp) / "missing.so"))

        self.assertFalse(status["installed"])
        self.assertFalse(status["loadable"])
        self.assertIn("not installed", str(status["error"]))

    def test_bridge_abi_version_mismatch_is_rejected(self) -> None:
        class Version:
            argtypes = None
            restype = None

            def __call__(self) -> int:
                return 1

        error = _bridge_abi_error(
            SimpleNamespace(osagent_kylin_vector_abi_version=Version())  # type: ignore[arg-type]
        )

        self.assertEqual(error, "bridge ABI version mismatch: expected 2, got 1")

    def test_search_passes_allowed_keys_through_abi(self) -> None:
        payload = ctypes.create_string_buffer(b"7\t0.9\tallowed-key\n")
        observed: list[str] = []

        def search(
            handle,  # type: ignore[no-untyped-def]
            values,  # type: ignore[no-untyped-def]
            dimension,  # type: ignore[no-untyped-def]
            limit,  # type: ignore[no-untyped-def]
            allowed,  # type: ignore[no-untyped-def]
            allowed_count,  # type: ignore[no-untyped-def]
            output,  # type: ignore[no-untyped-def]
            output_size,  # type: ignore[no-untyped-def]
            error,  # type: ignore[no-untyped-def]
            error_size,  # type: ignore[no-untyped-def]
        ) -> int:
            del handle, values, dimension, limit, error, error_size
            observed.extend(allowed[index].decode("utf-8") for index in range(allowed_count))
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.cast(
                payload,
                ctypes.c_void_p,
            )
            ctypes.cast(output_size, ctypes.POINTER(ctypes.c_size_t))[0] = len(
                payload.value
            )
            return 0

        engine = object.__new__(KylinVectorEngine)
        engine._handle = ctypes.c_void_p(1)
        engine._keys = set()
        engine._library = SimpleNamespace(  # type: ignore[assignment]
            osagent_kylin_vector_search=search,
            osagent_kylin_vector_free=lambda output: None,
        )

        hits = engine.search([1.0, 0.0], 1, allowed_keys={"allowed-key"})

        self.assertEqual(observed, ["allowed-key"])
        self.assertEqual(hits, [("allowed-key", 0.9)])

    def test_collection_name_is_scoped_by_embedding_dimension(self) -> None:
        with TemporaryDirectory() as tmp, mock.patch.object(
            KylinVectorEngine,
            "__init__",
            return_value=None,
        ) as initialize:
            KylinVectorEngine.from_environment(
                Path(tmp),
                kind="facts",
                dimension=768,
            )

        self.assertEqual(
            initialize.call_args.kwargs["collection"],
            "osagent_memory_facts_768d",
        )


if __name__ == "__main__":
    unittest.main()
