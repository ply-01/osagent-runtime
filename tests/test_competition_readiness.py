from __future__ import annotations

import ctypes
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from os_agent import competition
from os_agent.memory.embedding_cache import EmbeddingCache
from os_agent.memory.embedding_provider import (
    EmbeddingConfig,
    EmbeddingError,
    KylinSystemEmbeddingProvider,
    create_embedding_provider,
)
from os_agent.memory.fact_vector_index import FactSemanticIndex
from os_agent.memory.mem0_schemas import FactCard


class _FakeFunction:
    def __init__(self, implementation):  # type: ignore[no-untyped-def]
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):  # type: ignore[no-untyped-def]
        return self.implementation(*args)


class _FakeKylinEmbeddingLibrary:
    def __init__(self, *, error_code: int = 0) -> None:
        self.error_code = error_code
        self.created_sessions = 0
        self.inputs: list[str] = []
        self.models: list[str] = []
        self.destroyed_results = 0
        self.destroyed_sessions = 0
        self.vector = (ctypes.c_float * 3)(0.25, -0.5, 0.75)

        self.text_embedding_create_session = _FakeFunction(self._create_session)
        self.text_embedding_destroy_session = _FakeFunction(self._destroy_session)
        self.text_embedding_init_session = _FakeFunction(lambda session: 0)
        self.text_embedding_init_model = _FakeFunction(self._init_model)
        self.text_embedding = _FakeFunction(self._embed)
        self.embedding_result_get_vector_data = _FakeFunction(
            lambda result: self.vector
        )
        self.embedding_result_get_vector_length = _FakeFunction(lambda result: 3)
        self.embedding_result_get_error_code = _FakeFunction(
            lambda result: self.error_code
        )
        self.embedding_result_get_error_message = _FakeFunction(
            lambda result: b"mock failure" if self.error_code else b"Success"
        )
        self.embedding_result_destroy = _FakeFunction(self._destroy_result)

    def _create_session(self) -> int:
        self.created_sessions += 1
        return 100 + self.created_sessions

    def _init_model(self, session, model: bytes) -> int:  # type: ignore[no-untyped-def]
        self.models.append(model.decode("utf-8"))
        return 0

    def _embed(self, session, text: bytes, result) -> bool:  # type: ignore[no-untyped-def]
        self.inputs.append(text.decode("utf-8"))
        result._obj.value = 202
        return self.error_code == 0

    def _destroy_result(self, result) -> None:  # type: ignore[no-untyped-def]
        self.destroyed_results += 1
        result._obj.value = None

    def _destroy_session(self, session) -> None:  # type: ignore[no-untyped-def]
        self.destroyed_sessions += 1
        session._obj.value = None


class _TransientTimeoutKylinEmbeddingLibrary(_FakeKylinEmbeddingLibrary):
    def __init__(self, *, persistent: bool = False) -> None:
        super().__init__()
        self.persistent = persistent
        self.text_embedding = _FakeFunction(self._embed_with_timeout)

    def _embed_with_timeout(
        self, session, text: bytes, result
    ) -> bool:  # type: ignore[no-untyped-def]
        self.inputs.append(text.decode("utf-8"))
        result._obj.value = 202
        self.error_code = 3 if self.persistent or len(self.inputs) == 1 else 0
        return self.error_code == 0


class _FakeKylinV11EmbeddingLibrary(_FakeKylinEmbeddingLibrary):
    def __init__(self, *, model: str = "ensemble-embd_gte-base_uint8-text") -> None:
        super().__init__()
        del self.text_embedding_init_model
        self.model = model
        self.destroyed_model_infos = 0
        self.text_embedding_get_model_info = _FakeFunction(self._get_model_info)
        self.embedding_model_info_destroy = _FakeFunction(
            self._destroy_model_info
        )

    def _get_model_info(self, session, result) -> bool:  # type: ignore[no-untyped-def]
        result._obj.value = json.dumps(
            {"dim": 3, "name": self.model, "ondevice": True}
        ).encode("utf-8")
        return True

    def _destroy_model_info(self, result) -> None:  # type: ignore[no-untyped-def]
        self.destroyed_model_infos += 1


class KylinSystemEmbeddingProviderTests(unittest.TestCase):
    def test_factory_selects_kylin_system_provider(self) -> None:
        provider = create_embedding_provider(
            EmbeddingConfig(backend="kylin_system", model="system-model")
        )

        self.assertIsInstance(provider, KylinSystemEmbeddingProvider)
        self.assertTrue(provider.enabled)
        self.assertIn("system-model", provider.cache_namespace)

    def test_provider_uses_sdk_and_releases_owned_resources(self) -> None:
        library = _FakeKylinEmbeddingLibrary()
        provider = KylinSystemEmbeddingProvider(
            EmbeddingConfig(
                backend="kylin_system",
                model="ensemble-embd_gte-base_uint8-text",
                library_path="fake-kylin-embedding.so",
            ),
            library_loader=lambda path: library,
        )

        vectors = provider.embed_documents(["银河麒麟", "cross-session memory"])
        provider.close()

        self.assertEqual(
            vectors,
            [[0.25, -0.5, 0.75], [0.25, -0.5, 0.75]],
        )
        self.assertEqual(library.inputs, ["银河麒麟", "cross-session memory"])
        self.assertEqual(
            library.models,
            ["ensemble-embd_gte-base_uint8-text"],
        )
        self.assertEqual(library.destroyed_results, 2)
        self.assertEqual(library.destroyed_sessions, 1)

    def test_provider_releases_failed_embedding_result(self) -> None:
        library = _FakeKylinEmbeddingLibrary(error_code=17)
        provider = KylinSystemEmbeddingProvider(
            EmbeddingConfig(
                backend="kylin_system",
                library_path="fake-kylin-embedding.so",
            ),
            library_loader=lambda path: library,
        )

        with self.assertRaisesRegex(EmbeddingError, "error code 17"):
            provider.embed_queries(["failure path"])
        provider.close()

        self.assertEqual(library.destroyed_results, 1)
        self.assertEqual(library.destroyed_sessions, 1)

    def test_provider_recreates_session_once_after_transient_timeout(self) -> None:
        library = _TransientTimeoutKylinEmbeddingLibrary()
        provider = KylinSystemEmbeddingProvider(
            EmbeddingConfig(
                backend="kylin_system",
                library_path="fake-kylin-embedding.so",
            ),
            library_loader=lambda path: library,
        )

        self.assertEqual(provider.embed_queries(["transient request"]), [[0.25, -0.5, 0.75]])
        provider.close()

        self.assertEqual(library.inputs, ["transient request", "transient request"])
        self.assertEqual(library.created_sessions, 2)
        self.assertEqual(library.destroyed_results, 2)
        self.assertEqual(library.destroyed_sessions, 2)

    def test_provider_fails_after_one_transient_timeout_retry(self) -> None:
        library = _TransientTimeoutKylinEmbeddingLibrary(persistent=True)
        provider = KylinSystemEmbeddingProvider(
            EmbeddingConfig(
                backend="kylin_system",
                library_path="fake-kylin-embedding.so",
            ),
            library_loader=lambda path: library,
        )

        with self.assertRaisesRegex(EmbeddingError, "error code 3"):
            provider.embed_queries(["persistent timeout"])
        provider.close()

        self.assertEqual(library.inputs, ["persistent timeout", "persistent timeout"])
        self.assertEqual(library.created_sessions, 2)
        self.assertEqual(library.destroyed_results, 2)
        self.assertEqual(library.destroyed_sessions, 2)

    def test_provider_supports_v11_model_info_abi(self) -> None:
        library = _FakeKylinV11EmbeddingLibrary()
        provider = KylinSystemEmbeddingProvider(
            EmbeddingConfig(
                backend="kylin_system",
                model="ensemble-embd_gte-base_uint8-text",
                library_path="fake-kylin-embedding.so",
            ),
            library_loader=lambda path: library,
        )

        self.assertEqual(provider.embed_queries(["V11"]), [[0.25, -0.5, 0.75]])
        provider.close()

        self.assertEqual(library.models, [])
        self.assertEqual(library.destroyed_model_infos, 1)
        self.assertEqual(library.destroyed_sessions, 1)

    def test_provider_rejects_v11_runtime_model_mismatch(self) -> None:
        library = _FakeKylinV11EmbeddingLibrary(model="different-model")
        provider = KylinSystemEmbeddingProvider(
            EmbeddingConfig(
                backend="kylin_system",
                model="configured-model",
                library_path="fake-kylin-embedding.so",
            ),
            library_loader=lambda path: library,
        )

        with self.assertRaisesRegex(EmbeddingError, "runtime model mismatch"):
            provider.embed_queries(["V11"])

        self.assertEqual(library.destroyed_model_infos, 1)
        self.assertEqual(library.destroyed_sessions, 1)

    def test_provider_rejects_an_abi_incompatible_library(self) -> None:
        provider = KylinSystemEmbeddingProvider(
            EmbeddingConfig(
                backend="kylin_system",
                library_path="wrong-library.so",
            ),
            library_loader=lambda path: object(),
        )

        with self.assertRaisesRegex(EmbeddingError, "ABI-incompatible"):
            provider.embed_queries(["probe"])

    def test_required_system_provider_does_not_silently_disable_semantic_index(
        self,
    ) -> None:
        library = _FakeKylinEmbeddingLibrary(error_code=17)
        provider = KylinSystemEmbeddingProvider(
            EmbeddingConfig(
                backend="kylin_system",
                library_path="fake-kylin-embedding.so",
            ),
            library_loader=lambda path: library,
        )
        fact = FactCard(
            entity="project",
            attribute="format",
            value="Markdown",
            evidence="Use Markdown for project reports.",
            source_session_id="session-a",
            source_event_id="event-a",
        )
        with TemporaryDirectory() as tmp:
            index = FactSemanticIndex(
                provider=provider,
                cache=EmbeddingCache(Path(tmp) / "embeddings.json"),
            )

            with self.assertRaisesRegex(EmbeddingError, "error code 17"):
                index.refresh([fact])
        provider.close()


class CompetitionPreflightTests(unittest.TestCase):
    def test_preflight_keeps_vector_integration_as_a_blocker(self) -> None:
        with TemporaryDirectory() as tmp:
            os_release = Path(tmp) / "os-release"
            os_release.write_text(
                'NAME="Galaxy Kylin Desktop"\n'
                'VERSION_ID="V11"\n'
                'PRETTY_NAME="Galaxy Kylin Desktop V11"\n',
                encoding="utf-8",
            )
            config = EmbeddingConfig(
                backend="kylin_system",
                model="system-model",
            )
            real_exists = Path.exists

            def fake_exists(path: Path) -> bool:
                if str(path).endswith(("core-textembedding.sock", ".sock")):
                    return True
                return real_exists(path)

            with (
                mock.patch.object(competition.platform, "system", return_value="Linux"),
                mock.patch.object(
                    competition,
                    "_first_loadable_library",
                    side_effect=["libembedding.so.1", "libvector.so.1"],
                ),
                mock.patch.object(Path, "exists", fake_exists),
            ):
                report = competition.run_competition_preflight(
                    embedding_config=config,
                    os_release_path=os_release,
                )

        failed = [check.check_id for check in report.checks if check.status == "fail"]
        self.assertEqual(failed, ["system_vector_integration"])
        self.assertFalse(report.ready)
        self.assertEqual(json.loads(report.to_json())["required_passed"], 6)

    def test_os_release_parser_ignores_comments_and_quotes(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "os-release"
            path.write_text(
                '# comment\nNAME="Kylin Desktop"\nVERSION_ID=\'V11\'\n',
                encoding="utf-8",
            )

            values = competition._read_os_release(path)

        self.assertEqual(values["NAME"], "Kylin Desktop")
        self.assertEqual(values["VERSION_ID"], "V11")

    def test_preflight_rejects_an_unloadable_vector_bridge(self) -> None:
        with TemporaryDirectory() as tmp:
            os_release = Path(tmp) / "os-release"
            bridge = Path(tmp) / "libbridge.so"
            os_release.write_text(
                'NAME="Galaxy Kylin Desktop"\n'
                'VERSION_ID="V11"\n'
                'PRETTY_NAME="Galaxy Kylin Desktop V11"\n',
                encoding="utf-8",
            )
            bridge.write_bytes(b"not-a-library")
            config = EmbeddingConfig(backend="kylin_system", model="system-model")

            with (
                mock.patch.object(competition.platform, "system", return_value="Linux"),
                mock.patch.object(
                    competition,
                    "_first_loadable_library",
                    side_effect=["libembedding.so.1", "libvector.so.1"],
                ),
                mock.patch.object(Path, "exists", return_value=True),
                mock.patch.object(
                    competition,
                    "kylin_vector_bridge_path",
                    return_value=bridge,
                ),
                mock.patch.object(
                    competition,
                    "kylin_vector_bridge_load_error",
                    return_value="OSError: undefined symbol: VectorDB::IndexDesc",
                ),
            ):
                report = competition.run_competition_preflight(
                    embedding_config=config,
                    os_release_path=os_release,
                )

        integration = next(
            check
            for check in report.checks
            if check.check_id == "system_vector_integration"
        )
        self.assertEqual(integration.status, "fail")
        self.assertIn("undefined symbol", integration.detail)
        self.assertFalse(report.ready)


class _FakeSmokeProvider:
    def __init__(self) -> None:
        self.closed = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [0.0, 1.0]]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0]]

    def close(self) -> None:
        self.closed = True


class _FakeSmokeEngine:
    collection = "osagent_memory_competition_smoke_2d"

    def __init__(self, *, retain_after_delete: bool = False) -> None:
        self.rows: dict[str, list[float]] = {}
        self.closed = False
        self.retain_after_delete = retain_after_delete

    def replace(self, documents) -> int:  # type: ignore[no-untyped-def]
        rows = list(documents)
        if rows or not self.retain_after_delete:
            self.rows = {key: vector for key, vector in rows}
        return len(rows)

    def search(
        self,
        query: list[float],
        limit: int,
        *,
        allowed_keys=None,  # type: ignore[no-untyped-def]
    ) -> list[tuple[str, float]]:
        allowed = set(self.rows) if allowed_keys is None else set(allowed_keys)
        hits = [
            (key, sum(left * right for left, right in zip(query, vector)))
            for key, vector in self.rows.items()
            if key in allowed
        ]
        return sorted(hits, key=lambda item: (-item[1], item[0]))[:limit]

    def close(self) -> None:
        self.closed = True


class CompetitionVectorSmokeTests(unittest.TestCase):
    def test_real_sdk_smoke_contract_writes_searches_and_deletes(self) -> None:
        provider = _FakeSmokeProvider()
        engine = _FakeSmokeEngine()
        config = EmbeddingConfig(
            backend="kylin_system",
            model="system-model",
        )
        with (
            mock.patch.object(
                competition,
                "create_embedding_provider",
                return_value=provider,
            ),
            mock.patch.object(
                competition.KylinVectorEngine,
                "from_environment",
                return_value=engine,
            ) as create_engine,
        ):
            report = competition.run_competition_vector_smoke(
                embedding_config=config,
                root="memory-root",
            )

        self.assertTrue(report.passed)
        self.assertEqual(report.dimension, 2)
        self.assertEqual(report.documents_written, 2)
        self.assertEqual(report.top_key, "competition-smoke-primary")
        self.assertEqual(report.filtered_top_key, "competition-smoke-secondary")
        self.assertEqual(report.updated_top_key, "competition-smoke-secondary")
        self.assertTrue(report.filter_verified)
        self.assertTrue(report.update_verified)
        self.assertTrue(report.delete_verified)
        self.assertTrue(provider.closed)
        self.assertTrue(engine.closed)
        create_engine.assert_called_once_with(
            Path("memory-root"),
            kind="competition_smoke",
            dimension=2,
        )

    def test_smoke_fails_when_native_delete_is_not_observed(self) -> None:
        provider = _FakeSmokeProvider()
        engine = _FakeSmokeEngine(retain_after_delete=True)
        with (
            mock.patch.object(
                competition,
                "create_embedding_provider",
                return_value=provider,
            ),
            mock.patch.object(
                competition.KylinVectorEngine,
                "from_environment",
                return_value=engine,
            ),
        ):
            report = competition.run_competition_vector_smoke(
                embedding_config=EmbeddingConfig(backend="kylin_system")
            )

        self.assertFalse(report.passed)
        self.assertFalse(report.delete_verified)

    def test_smoke_refuses_non_system_embedding_backend(self) -> None:
        report = competition.run_competition_vector_smoke(
            embedding_config=EmbeddingConfig(backend="openai")
        )

        self.assertFalse(report.passed)
        self.assertIn("kylin_system", report.error)


if __name__ == "__main__":
    unittest.main()
