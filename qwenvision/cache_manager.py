from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM


@dataclass
class QwenVisionModelHandle:
    model: object
    processor: object
    model_id: str
    device: str
    dtype: str
    cache_key: str


class ModelCacheManager:
    def __init__(self) -> None:
        self._cache: Dict[str, dict] = {}
        self._lock = Lock()

    def _make_cache_key(self, model_id: str, device: str, dtype: str) -> str:
        return f"{model_id}|{device}|{dtype}"

    def _resolve_dtype(self, dtype: str):
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        if dtype == "float32":
            return torch.float32
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def _resolve_device_map(self, device: str):
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            return "cuda"
        return "auto"

    def _load_model_and_processor(self, model_id: str, device: str, dtype: str):
        torch_dtype = self._resolve_dtype(dtype)
        device_map = self._resolve_device_map(device)

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        model = None
        load_errors = []

        # Prefer a multimodal-specific auto class first.
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        except Exception as e:
            load_errors.append(f"AutoModelForImageTextToText failed: {e}")

        # Fall back for models that expose multimodal support through CausalLM.
        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            except Exception as e:
                load_errors.append(f"AutoModelForCausalLM failed: {e}")

        if model is None:
            joined = " | ".join(load_errors)
            raise RuntimeError(f"Could not load model '{model_id}'. Details: {joined}")

        return model, processor, str(getattr(model, "device", device_map)), str(torch_dtype)

    def get_or_load_model(self, model_id: str, device: str = "auto", dtype: str = "auto") -> QwenVisionModelHandle:
        key = self._make_cache_key(model_id, device, dtype)

        with self._lock:
            if key in self._cache:
                self._cache[key]["last_used"] = time.time()
                entry = self._cache[key]
                return QwenVisionModelHandle(
                    model=entry["model"],
                    processor=entry["processor"],
                    model_id=entry["model_id"],
                    device=entry["device"],
                    dtype=entry["dtype"],
                    cache_key=key,
                )

            model, processor, resolved_device, resolved_dtype = self._load_model_and_processor(
                model_id=model_id,
                device=device,
                dtype=dtype,
            )

            self._cache[key] = {
                "model": model,
                "processor": processor,
                "model_id": model_id,
                "device": resolved_device,
                "dtype": resolved_dtype,
                "last_used": time.time(),
            }

            return QwenVisionModelHandle(
                model=model,
                processor=processor,
                model_id=model_id,
                device=resolved_device,
                dtype=resolved_dtype,
                cache_key=key,
            )

    def unload_by_key(self, cache_key: str) -> None:
        with self._lock:
            entry = self._cache.pop(cache_key, None)
            if entry is not None:
                del entry
        self.clear_cuda_cache()

    def unload_all(self) -> None:
        with self._lock:
            keys = list(self._cache.keys())
            for key in keys:
                del self._cache[key]
            self._cache.clear()
        self.clear_cuda_cache()

    def clear_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_cache_status(self):
        with self._lock:
            return {
                key: {
                    "model_id": value["model_id"],
                    "device": value["device"],
                    "dtype": value["dtype"],
                    "last_used": value["last_used"],
                }
                for key, value in self._cache.items()
            }


_CACHE_MANAGER = ModelCacheManager()


def get_cache_manager() -> ModelCacheManager:
    return _CACHE_MANAGER
