"""Configurable remote and local embedding providers for memory retrieval."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
DEFAULT_E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DEFAULT_EMBEDDING_CONFIG_PATH = "config/embedding.local.json"


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot return vectors."""


class _KylinEmbeddingCallError(EmbeddingError):
    """Preserve the SDK error code for bounded transient recovery."""

    def __init__(self, error_code: int, detail: str) -> None:
        self.error_code = error_code
        super().__init__(
            f"Kylin embedding failed with error code {error_code}: {detail}"
        )


class EmbeddingProvider(Protocol):
    """Common interface used by all semantic memory indexes."""

    @property
    def enabled(self) -> bool: ...

    @property
    def cache_namespace(self) -> str: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class EmbeddingConfig:
    """Runtime configuration for remote or local embeddings."""

    backend: str = "openai"
    api_key: str | None = None
    base_url: str = DEFAULT_EMBEDDING_BASE_URL
    model: str | None = None
    revision: str | None = None
    timeout_seconds: int = 60
    device: str = "auto"
    batch_size: int = 16
    max_length: int = 512
    local_files_only: bool = False
    onnx_filename: str = "onnx/model_qint8_avx512_vnni.onnx"
    onnx_path: str | None = None
    onnx_threads: int = 8
    library_path: str | None = None

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        file_config = cls._from_local_file()
        model = (
            os.getenv("OS_AGENT_EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or file_config.get("embedding_model")
        )
        revision = (
            os.getenv("OS_AGENT_EMBEDDING_REVISION")
            or file_config.get("embedding_revision")
        )
        backend = str(
            os.getenv("OS_AGENT_EMBEDDING_BACKEND")
            or file_config.get("embedding_backend")
            or "openai"
        )
        if (
            backend.lower().replace("-", "_")
            in {
                "local",
                "local_transformers",
                "transformers",
                "local_onnx",
                "onnx",
                "onnxruntime",
            }
            and model == "intfloat/multilingual-e5-small"
            and not revision
        ):
            revision = DEFAULT_E5_REVISION
        return cls(
            backend=backend,
            api_key=(
                os.getenv("OS_AGENT_EMBEDDING_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("OS_AGENT_API_KEY")
                or file_config.get("embedding_api_key")
                or file_config.get("api_key")
            ),
            base_url=(
                os.getenv("OS_AGENT_EMBEDDING_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OS_AGENT_BASE_URL")
                or str(
                    file_config.get("embedding_base_url")
                    or file_config.get("base_url")
                    or DEFAULT_EMBEDDING_BASE_URL
                )
            ),
            model=str(model) if model else None,
            revision=str(revision) if revision else None,
            timeout_seconds=int(
                os.getenv("OS_AGENT_EMBEDDING_TIMEOUT_SECONDS")
                or file_config.get("embedding_timeout_seconds")
                or file_config.get("timeout_seconds")
                or "60"
            ),
            device=str(
                os.getenv("OS_AGENT_EMBEDDING_DEVICE")
                or file_config.get("embedding_device")
                or "auto"
            ),
            batch_size=max(
                int(
                    os.getenv("OS_AGENT_EMBEDDING_BATCH_SIZE")
                    or file_config.get("embedding_batch_size")
                    or "16"
                ),
                1,
            ),
            max_length=max(
                int(
                    os.getenv("OS_AGENT_EMBEDDING_MAX_LENGTH")
                    or file_config.get("embedding_max_length")
                    or "512"
                ),
                1,
            ),
            local_files_only=_as_bool(
                os.getenv("OS_AGENT_EMBEDDING_LOCAL_FILES_ONLY")
                or file_config.get("embedding_local_files_only"),
                default=False,
            ),
            onnx_filename=str(
                os.getenv("OS_AGENT_EMBEDDING_ONNX_FILENAME")
                or file_config.get("embedding_onnx_filename")
                or "onnx/model_qint8_avx512_vnni.onnx"
            ),
            onnx_path=(
                os.getenv("OS_AGENT_EMBEDDING_ONNX_PATH")
                or file_config.get("embedding_onnx_path")
            ),
            onnx_threads=max(
                int(
                    os.getenv("OS_AGENT_EMBEDDING_ONNX_THREADS")
                    or file_config.get("embedding_onnx_threads")
                    or "8"
                ),
                0,
            ),
            library_path=(
                os.getenv("OS_AGENT_EMBEDDING_LIBRARY_PATH")
                or file_config.get("embedding_library_path")
            ),
        )

    @classmethod
    def _from_local_file(cls) -> dict[str, Any]:
        embedding_config = os.getenv("OS_AGENT_EMBEDDING_CONFIG")
        if embedding_config:
            return cls._read_config(Path(embedding_config))
        dedicated = cls._read_config(Path(DEFAULT_EMBEDDING_CONFIG_PATH))
        if dedicated:
            return dedicated
        return cls._read_config(
            Path(os.getenv("OS_AGENT_LLM_CONFIG", "config/llm.local.json"))
        )

    @classmethod
    def _read_config(cls, path: Path) -> dict[str, Any]:
        if not path.is_absolute() and not path.exists():
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            candidate = project_root / path
            if candidate.exists():
                path = candidate
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"Invalid embedding config file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise EmbeddingError(f"Invalid embedding config file {path}: root must be an object")
        return data


class OpenAICompatibleEmbeddingProvider:
    """Small embeddings client with no third-party dependency."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig.from_env()

    @property
    def enabled(self) -> bool:
        return bool(self.config.api_key and self.config.model)

    @property
    def cache_namespace(self) -> str:
        return f"openai|{self.config.base_url.rstrip('/')}|{self.config.model or ''}"

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.enabled:
            raise EmbeddingError("embedding api key or model is not configured")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": texts,
        }
        body = json.dumps(payload).encode("utf-8")
        url = self.config.base_url.rstrip("/") + "/embeddings"
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingError(f"Embedding HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(f"Embedding connection error: {exc}") from exc
        except TimeoutError as exc:
            raise EmbeddingError("Embedding request timed out") from exc

        try:
            rows = sorted(data["data"], key=lambda item: int(item.get("index", 0)))
            return [[float(value) for value in row["embedding"]] for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"Unexpected embedding response: {data}") from exc


class KylinSystemEmbeddingProvider:
    """Galaxy Kylin text embeddings through the system SDK C ABI."""

    _TRANSIENT_ERROR_CODES = frozenset({3})
    _LIBRARY_NAMES = (
        "libkysdk-coreai-embedding.so.1",
        "libkysdk-coreai-embedding.so",
        "kysdk-coreai-embedding",
    )

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        library_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig.from_env()
        self._library_loader = library_loader or ctypes.CDLL
        self._library: Any = None
        self._session: ctypes.c_void_p | None = None
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        # Selecting this backend is explicit. Runtime failures must surface instead
        # of silently disabling semantic retrieval or switching providers.
        return True

    @property
    def is_kylin_system(self) -> bool:
        """Marker used to enable the mandatory Kylin vector-engine path."""
        return True

    @property
    def required(self) -> bool:
        return True

    @property
    def cache_namespace(self) -> str:
        model = self.config.model or "system-default"
        return f"kylin-system|abi=1|model={model}"

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def close(self) -> None:
        with self._lock:
            if self._session is None or self._library is None:
                return
            session = self._session
            self._session = None
            self._library.text_embedding_destroy_session(ctypes.byref(session))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            self._ensure_session()
            vectors: list[list[float]] = []
            for text in texts:
                try:
                    vector = self._embed_one(text)
                except _KylinEmbeddingCallError as exc:
                    if exc.error_code not in self._TRANSIENT_ERROR_CODES:
                        raise
                    self.close()
                    self._ensure_session()
                    vector = self._embed_one(text)
                vectors.append(vector)
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or 0 in dimensions:
            raise EmbeddingError(
                "Kylin embedding SDK returned inconsistent vector dimensions"
            )
        return vectors

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        library = self._load_library()
        try:
            self._configure_abi(library)
        except AttributeError as exc:
            raise EmbeddingError(
                "Kylin embedding library is ABI-incompatible: "
                f"missing required symbol {exc}"
            ) from exc
        raw_session = library.text_embedding_create_session()
        if not raw_session:
            raise EmbeddingError("Kylin embedding SDK could not create a session")
        session = (
            raw_session
            if isinstance(raw_session, ctypes.c_void_p)
            else ctypes.c_void_p(raw_session)
        )
        try:
            error_code = int(library.text_embedding_init_session(session))
            if error_code != 0:
                raise EmbeddingError(
                    "Kylin embedding runtime initialization failed "
                    f"with error code {error_code}"
                )
            init_model = getattr(library, "text_embedding_init_model", None)
            if self.config.model and init_model is not None:
                error_code = int(
                    init_model(session, self.config.model.encode("utf-8"))
                )
                if error_code != 0:
                    raise EmbeddingError(
                        f"Kylin embedding model {self.config.model!r} failed to load "
                        f"with error code {error_code}"
                    )
            elif self.config.model:
                model_info = self._get_model_info(library, session)
                actual_model = str(model_info.get("name") or "")
                if not actual_model:
                    raise EmbeddingError(
                        "Kylin embedding SDK did not report the active model"
                    )
                if actual_model != self.config.model:
                    raise EmbeddingError(
                        "Kylin embedding runtime model mismatch: configured "
                        f"{self.config.model!r}, active {actual_model!r}"
                    )
        except Exception:
            library.text_embedding_destroy_session(ctypes.byref(session))
            raise
        self._library = library
        self._session = session

    @staticmethod
    def _get_model_info(library: Any, session: ctypes.c_void_p) -> dict[str, Any]:
        get_model_info = getattr(library, "text_embedding_get_model_info", None)
        destroy_model_info = getattr(library, "embedding_model_info_destroy", None)
        if get_model_info is None or destroy_model_info is None:
            raise EmbeddingError(
                "Kylin embedding library cannot select or report the active model"
            )
        result = ctypes.c_char_p()
        try:
            if (
                not bool(get_model_info(session, ctypes.byref(result)))
                or not result.value
            ):
                raise EmbeddingError(
                    "Kylin embedding SDK could not report the active model"
                )
            try:
                model_info = json.loads(result.value.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EmbeddingError(
                    "Kylin embedding SDK returned invalid model information"
                ) from exc
            if not isinstance(model_info, dict):
                raise EmbeddingError(
                    "Kylin embedding SDK returned invalid model information"
                )
            return model_info
        finally:
            if result.value:
                destroy_model_info(result)

    def _embed_one(self, text: str) -> list[float]:
        if self._library is None or self._session is None:
            raise EmbeddingError("Kylin embedding session is not initialized")
        result = ctypes.c_void_p()
        try:
            success = bool(
                self._library.text_embedding(
                    self._session,
                    text.encode("utf-8"),
                    ctypes.byref(result),
                )
            )
            if not result.value:
                raise EmbeddingError("Kylin embedding SDK returned no result")
            error_code = int(
                self._library.embedding_result_get_error_code(result)
            )
            if not success or error_code != 0:
                message = self._library.embedding_result_get_error_message(result)
                detail = (
                    message.decode("utf-8", errors="replace")
                    if message
                    else "unknown error"
                )
                raise _KylinEmbeddingCallError(error_code, detail)
            length = int(self._library.embedding_result_get_vector_length(result))
            data = self._library.embedding_result_get_vector_data(result)
            if length <= 0 or not data:
                raise EmbeddingError("Kylin embedding SDK returned an empty vector")
            return [float(data[index]) for index in range(length)]
        finally:
            if result.value:
                self._library.embedding_result_destroy(ctypes.byref(result))

    def _load_library(self) -> Any:
        failures: list[str] = []
        for candidate in kylin_embedding_library_candidates(self.config):
            try:
                return self._library_loader(candidate)
            except OSError as exc:
                failures.append(f"{candidate}: {exc}")
        detail = "; ".join(failures) if failures else "no library candidate found"
        raise EmbeddingError(
            "Kylin system embedding library is unavailable; install "
            f"libkylin-coreai-embedding or set OS_AGENT_EMBEDDING_LIBRARY_PATH ({detail})"
        )

    @staticmethod
    def _configure_abi(library: Any) -> None:
        library.text_embedding_create_session.argtypes = []
        library.text_embedding_create_session.restype = ctypes.c_void_p
        library.text_embedding_destroy_session.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        library.text_embedding_destroy_session.restype = None
        library.text_embedding_init_session.argtypes = [ctypes.c_void_p]
        library.text_embedding_init_session.restype = ctypes.c_int
        init_model = getattr(library, "text_embedding_init_model", None)
        if init_model is not None:
            init_model.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
            ]
            init_model.restype = ctypes.c_int
        get_model_info = getattr(library, "text_embedding_get_model_info", None)
        destroy_model_info = getattr(library, "embedding_model_info_destroy", None)
        if get_model_info is not None and destroy_model_info is not None:
            get_model_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_char_p),
            ]
            get_model_info.restype = ctypes.c_bool
            destroy_model_info.argtypes = [ctypes.c_char_p]
            destroy_model_info.restype = None
        library.text_embedding.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.text_embedding.restype = ctypes.c_bool
        library.embedding_result_get_vector_data.argtypes = [ctypes.c_void_p]
        library.embedding_result_get_vector_data.restype = ctypes.POINTER(
            ctypes.c_float
        )
        library.embedding_result_get_vector_length.argtypes = [ctypes.c_void_p]
        library.embedding_result_get_vector_length.restype = ctypes.c_int
        library.embedding_result_get_error_code.argtypes = [ctypes.c_void_p]
        library.embedding_result_get_error_code.restype = ctypes.c_int
        library.embedding_result_get_error_message.argtypes = [ctypes.c_void_p]
        library.embedding_result_get_error_message.restype = ctypes.c_char_p
        library.embedding_result_destroy.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        library.embedding_result_destroy.restype = None

class LocalTransformerEmbeddingProvider:
    """Lazy local embedding provider implemented with torch and transformers."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig.from_env()
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = ""

    @property
    def enabled(self) -> bool:
        return bool(self.config.model)

    @property
    def cache_namespace(self) -> str:
        return "|".join(
            (
                "local-transformers",
                self.config.model or "",
                self.config.revision or "main",
                f"max_length={self.config.max_length}",
                f"pooling={self._pooling_strategy}",
            )
        )

    @property
    def _pooling_strategy(self) -> str:
        return "mean" if "e5" in (self.config.model or "").lower() else "cls"

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(self._prefix(texts, "query"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(self._prefix(texts, "passage"))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible document embedding method."""
        return self.embed_documents(texts)

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        if "e5" not in (self.config.model or "").lower():
            return texts
        prefix = f"{kind}: "
        return [text if text.startswith(prefix) else prefix + text for text in texts]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.enabled:
            raise EmbeddingError("local embedding model is not configured")
        self._ensure_model()
        vectors: list[list[float]] = []
        torch = self._torch
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            encoded = self._tokenizer(
                batch,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(self._device) for key, value in encoded.items()}
            try:
                with torch.inference_mode():
                    output = self._model(**model_inputs)
                    hidden = (
                        output.last_hidden_state
                        if hasattr(output, "last_hidden_state")
                        else output[0]
                    )
                    if self._pooling_strategy == "mean":
                        mask = model_inputs["attention_mask"].unsqueeze(-1).bool()
                        hidden = hidden.masked_fill(~mask, 0.0)
                        pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    else:
                        pooled = hidden[:, 0]
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            except Exception as exc:
                raise EmbeddingError(f"Local embedding inference failed: {exc}") from exc
            vectors.extend(normalized.detach().cpu().float().tolist())
        return vectors

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise EmbeddingError(
                "local embeddings require torch and transformers; "
                "install the 'local-embeddings' optional dependencies"
            ) from exc
        device = self.config.device.strip().lower()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
                trust_remote_code=False,
            )
            model = AutoModel.from_pretrained(
                self.config.model,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
                trust_remote_code=False,
            )
            model.eval()
            model.to(device)
        except Exception as exc:
            raise EmbeddingError(f"Unable to load local embedding model: {exc}") from exc
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device


class LocalOnnxEmbeddingProvider:
    """Quantized local E5 inference through ONNX Runtime."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig.from_env()
        self._numpy: Any = None
        self._tokenizer: Any = None
        self._session: Any = None
        self._input_names: set[str] = set()

    @property
    def enabled(self) -> bool:
        return bool(self.config.model)

    @property
    def cache_namespace(self) -> str:
        return "|".join(
            (
                "local-onnx",
                self.config.model or "",
                self.config.revision or "main",
                self.config.onnx_filename,
                f"max_length={self.config.max_length}",
                "pooling=mean",
            )
        )

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(self._prefix(texts, "query"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(self._prefix(texts, "passage"))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        if "e5" not in (self.config.model or "").lower():
            return texts
        prefix = f"{kind}: "
        return [text if text.startswith(prefix) else prefix + text for text in texts]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.enabled:
            raise EmbeddingError("local ONNX embedding model is not configured")
        self._ensure_model()
        np = self._numpy
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            encoded = self._tokenizer.encode_batch(batch)
            arrays = {
                "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": np.asarray(
                    [item.attention_mask for item in encoded], dtype=np.int64
                ),
                "token_type_ids": np.asarray(
                    [item.type_ids for item in encoded], dtype=np.int64
                ),
            }
            model_inputs = {
                name: arrays[name] for name in self._input_names if name in arrays
            }
            try:
                hidden = self._session.run(["last_hidden_state"], model_inputs)[0]
                mask = model_inputs["attention_mask"][..., None].astype(np.float32)
                pooled = (hidden * mask).sum(axis=1) / np.maximum(
                    mask.sum(axis=1), 1.0
                )
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                normalized = pooled / np.maximum(norms, 1e-12)
            except Exception as exc:
                raise EmbeddingError(f"Local ONNX embedding inference failed: {exc}") from exc
            vectors.extend(normalized.astype(np.float32).tolist())
        return vectors

    def _ensure_model(self) -> None:
        if self._session is not None:
            return
        try:
            import numpy as np
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download, try_to_load_from_cache
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EmbeddingError(
                "local ONNX embeddings require numpy, onnxruntime, tokenizers, "
                "and huggingface-hub"
            ) from exc
        try:
            model_path = self.config.onnx_path
            if not model_path:
                cached = try_to_load_from_cache(
                    self.config.model or "",
                    self.config.onnx_filename,
                    revision=self.config.revision,
                )
                model_path = cached if isinstance(cached, str) else None
            if not model_path and not self.config.local_files_only:
                model_path = hf_hub_download(
                    repo_id=self.config.model or "",
                    filename=self.config.onnx_filename,
                    revision=self.config.revision,
                )
            if not model_path:
                raise FileNotFoundError(self.config.onnx_filename)
            tokenizer_path = try_to_load_from_cache(
                self.config.model or "",
                "tokenizer.json",
                revision=self.config.revision,
            )
            if not isinstance(tokenizer_path, str) and not self.config.local_files_only:
                tokenizer_path = hf_hub_download(
                    repo_id=self.config.model or "",
                    filename="tokenizer.json",
                    revision=self.config.revision,
                )
            if not isinstance(tokenizer_path, str):
                raise FileNotFoundError("tokenizer.json")
            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer.enable_truncation(max_length=self.config.max_length)
            tokenizer.enable_padding(
                pad_id=tokenizer.token_to_id("<pad>") or 1,
                pad_token="<pad>",
            )
            options = ort.SessionOptions()
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
            if self.config.onnx_threads > 0:
                options.intra_op_num_threads = self.config.onnx_threads
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise EmbeddingError(f"Unable to load local ONNX embedding model: {exc}") from exc
        self._numpy = np
        self._tokenizer = tokenizer
        self._session = session
        self._input_names = {item.name for item in session.get_inputs()}


def create_embedding_provider(
    config: EmbeddingConfig | None = None,
) -> EmbeddingProvider:
    """Build the configured provider without loading a local model eagerly."""
    resolved = config or EmbeddingConfig.from_env()
    backend = resolved.backend.lower().replace("-", "_")
    if backend in {"local_onnx", "onnx", "onnxruntime"}:
        return LocalOnnxEmbeddingProvider(resolved)
    if backend in {"local", "local_transformers", "transformers"}:
        return LocalTransformerEmbeddingProvider(resolved)
    if backend in {"kylin", "kylin_system", "system_kylin"}:
        return KylinSystemEmbeddingProvider(resolved)
    if backend in {"openai", "openai_compatible", "remote"}:
        return OpenAICompatibleEmbeddingProvider(resolved)
    raise EmbeddingError(f"Unsupported embedding backend: {resolved.backend}")


def kylin_embedding_library_candidates(config: EmbeddingConfig) -> list[str]:
    """Return deterministic loader candidates for the Kylin SDK shared library."""
    if config.library_path:
        return [config.library_path]
    candidates: list[str] = []
    discovered = ctypes.util.find_library("kysdk-coreai-embedding")
    if discovered:
        candidates.append(discovered)
    candidates.extend(KylinSystemEmbeddingProvider._LIBRARY_NAMES)
    return list(dict.fromkeys(candidates))


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
