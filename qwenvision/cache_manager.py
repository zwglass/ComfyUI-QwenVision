from __future__ import annotations

import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict

from huggingface_hub import hf_hub_download


@dataclass
class QwenVisionModelHandle:
    model_path: str
    mmproj_path: str
    model_source: str
    mmproj_source: str
    cli_path: str
    cache_key: str


class ModelCacheManager:
    def __init__(self) -> None:
        self._cache: Dict[str, dict] = {}
        self._lock = Lock()

    def _make_cache_key(
        self, model_source: str, mmproj_source: str, cli_path: str
    ) -> str:
        return f"{model_source}|{mmproj_source}|{cli_path}"

    def _resolve_qwenvision_models_dir(self) -> str:
        try:
            import folder_paths

            models_dir = os.path.join(folder_paths.models_dir, "qwenvision")
        except Exception:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            comfyui_root = os.path.abspath(os.path.join(repo_root, "..", ".."))
            models_dir = os.path.join(comfyui_root, "models", "qwenvision")

        os.makedirs(models_dir, exist_ok=True)
        return models_dir

    def _parse_hf_resolve_url(self, source: str):
        # Example:
        # https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/xxx.gguf
        prefix = "https://huggingface.co/"
        if not source.startswith(prefix):
            return None

        parts = source[len(prefix) :].split("/")
        if len(parts) < 5:
            return None
        if parts[2] != "resolve":
            return None

        namespace = parts[0]
        repo_name = parts[1]
        revision = parts[3]
        filename = "/".join(parts[4:])
        repo_id = f"{namespace}/{repo_name}"
        return repo_id, revision, filename, repo_name

    def _resolve_source_to_local_file(self, source: str) -> str:
        if os.path.isfile(source):
            return source

        parsed = self._parse_hf_resolve_url(source)
        if parsed is None:
            raise RuntimeError(
                "Unsupported source. Use a local .gguf path or a huggingface "
                "resolve URL."
            )

        repo_id, revision, filename, repo_name = parsed
        local_dir = os.path.join(self._resolve_qwenvision_models_dir(), repo_name)
        os.makedirs(local_dir, exist_ok=True)

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

    def get_or_load_model(
        self,
        model_source: str,
        mmproj_source: str,
        cli_path: str = "llama-mtmd-cli",
    ) -> QwenVisionModelHandle:
        key = self._make_cache_key(model_source, mmproj_source, cli_path)

        with self._lock:
            if key in self._cache:
                self._cache[key]["last_used"] = time.time()
                entry = self._cache[key]
                return QwenVisionModelHandle(
                    model_path=entry["model_path"],
                    mmproj_path=entry["mmproj_path"],
                    model_source=entry["model_source"],
                    mmproj_source=entry["mmproj_source"],
                    cli_path=entry["cli_path"],
                    cache_key=key,
                )

            model_path = self._resolve_source_to_local_file(model_source)
            mmproj_path = self._resolve_source_to_local_file(mmproj_source)

            self._cache[key] = {
                "model_path": model_path,
                "mmproj_path": mmproj_path,
                "model_source": model_source,
                "mmproj_source": mmproj_source,
                "cli_path": cli_path,
                "last_used": time.time(),
            }

            return QwenVisionModelHandle(
                model_path=model_path,
                mmproj_path=mmproj_path,
                model_source=model_source,
                mmproj_source=mmproj_source,
                cli_path=cli_path,
                cache_key=key,
            )

    def unload_by_key(self, cache_key: str) -> None:
        with self._lock:
            self._cache.pop(cache_key, None)

    def unload_all(self) -> None:
        with self._lock:
            self._cache.clear()

    def get_cache_status(self):
        with self._lock:
            return {
                key: {
                    "model_path": value["model_path"],
                    "mmproj_path": value["mmproj_path"],
                    "model_source": value["model_source"],
                    "mmproj_source": value["mmproj_source"],
                    "cli_path": value["cli_path"],
                    "last_used": value["last_used"],
                }
                for key, value in self._cache.items()
            }


_CACHE_MANAGER = ModelCacheManager()


def get_cache_manager() -> ModelCacheManager:
    return _CACHE_MANAGER
