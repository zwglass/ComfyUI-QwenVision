from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoProcessor

from .managed_model import QwenVisionManagedModel

try:
    import comfy.model_management as mm
except Exception:
    mm = None

try:
    import comfy.model_patcher as mp
except Exception:
    mp = None


def _resolve_model_class(model_source: str):
    import transformers

    config = AutoConfig.from_pretrained(model_source)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_vl":
        if hasattr(transformers, "Qwen3VLForConditionalGeneration"):
            return transformers.Qwen3VLForConditionalGeneration
    if model_type == "qwen3_vl_moe":
        if hasattr(transformers, "Qwen3VLMoeForConditionalGeneration"):
            return transformers.Qwen3VLMoeForConditionalGeneration

    if hasattr(transformers, "AutoModelForImageTextToText"):
        return transformers.AutoModelForImageTextToText
    if hasattr(transformers, "AutoModelForVision2Seq"):
        return transformers.AutoModelForVision2Seq
    raise RuntimeError(
        "No supported vision-language model loader found in transformers "
        f"for model_type={model_type!r}."
    )


@dataclass
class QwenVisionModelHandle:
    model_source: str
    cache_key: str
    dtype: str
    device_map: str
    attn_implementation: str
    was_reused: bool = False


class ModelCacheManager:
    def __init__(self) -> None:
        self._cache: Dict[str, dict] = {}
        self._lock = Lock()

    def _resolve_target_device(self, device_map: str) -> torch.device:
        if device_map and device_map != "auto":
            return torch.device(device_map)
        if mm is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return mm.get_torch_device()
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_cache_key(
        self,
        model_source: str,
        dtype: str,
        device_map: str,
        attn_implementation: str,
    ) -> str:
        return (
            f"{model_source}|{dtype}|{device_map}|{attn_implementation}"
        )

    def _load_runtime(
        self,
        model_source: str,
        dtype: str,
        device_map: str,
        attn_implementation: str,
    ) -> tuple[Any, Any, Any, str]:
        model_class = _resolve_model_class(model_source)
        torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
        target_device = self._resolve_target_device(device_map)
        load_kwargs = {"torch_dtype": torch_dtype}
        if attn_implementation != "default":
            load_kwargs["attn_implementation"] = attn_implementation

        hf_model = model_class.from_pretrained(model_source, **load_kwargs)
        hf_model.eval()
        model = QwenVisionManagedModel(hf_model, initial_device=torch.device("cpu"))
        if mm is not None:
            mm.archive_model_dtypes(model)
        processor = AutoProcessor.from_pretrained(model_source)

        if mp is None:
            return model, processor, None, str(target_device)

        offload_device = target_device if target_device.type == "cpu" else torch.device("cpu")
        patcher = mp.ModelPatcher(
            model,
            load_device=target_device,
            offload_device=offload_device,
        )
        return model, processor, patcher, str(target_device)

    def _make_handle(
        self,
        entry: dict,
        cache_key: str,
        *,
        was_reused: bool,
    ) -> QwenVisionModelHandle:
        return QwenVisionModelHandle(
            model_source=entry["model_source"],
            cache_key=cache_key,
            dtype=entry["dtype"],
            device_map=entry["device_map"],
            attn_implementation=entry["attn_implementation"],
            was_reused=was_reused,
        )

    def get_or_load_model(
        self,
        model_source: str,
        dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: str = "default",
    ) -> QwenVisionModelHandle:
        key = self._make_cache_key(
            model_source=model_source,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry["last_used"] = time.time()
                return self._make_handle(entry, key, was_reused=True)

            model, processor, patcher, resolved_device = self._load_runtime(
                model_source=model_source,
                dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
            self._cache[key] = {
                "model_source": model_source,
                "model": model,
                "processor": processor,
                "patcher": patcher,
                "dtype": dtype,
                "device_map": resolved_device,
                "attn_implementation": attn_implementation,
                "last_used": time.time(),
            }
            return self._make_handle(self._cache[key], key, was_reused=False)

    def get_runtime_by_key(self, cache_key: str) -> tuple[Any, Any, Any, dict]:
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                raise RuntimeError(
                    "Model runtime was not found in cache. Load the model again."
                )
            model = entry.get("model")
            processor = entry.get("processor")
            patcher = entry.get("patcher")
            if model is None or processor is None:
                raise RuntimeError(
                    "Model runtime has been unloaded. Load the model again."
                )
            entry["last_used"] = time.time()
            metadata = {
                "model_source": entry["model_source"],
                "dtype": entry["dtype"],
                "device_map": entry["device_map"],
                "attn_implementation": entry["attn_implementation"],
            }
            return model, processor, patcher, metadata

    def _unload_patcher(self, patcher: Any) -> None:
        if patcher is None:
            return

        unloaded = False
        if mm is not None:
            try:
                for i in range(len(mm.current_loaded_models) - 1, -1, -1):
                    loaded_model = mm.current_loaded_models[i]
                    if loaded_model.model is not patcher:
                        continue
                    loaded_model.model_unload()
                    mm.current_loaded_models.pop(i)
                    unloaded = True
            except Exception:
                unloaded = False

        if not unloaded:
            try:
                patcher.detach(unpatch_all=True)
            except Exception:
                pass

    def _dispose_entry(self, entry: dict) -> None:
        model = entry.pop("model", None)
        processor = entry.pop("processor", None)
        patcher = entry.pop("patcher", None)
        entry["model"] = None
        entry["processor"] = None
        entry["patcher"] = None
        self._unload_patcher(patcher)
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        if model is not None:
            del model
        if processor is not None:
            del processor
        if patcher is not None:
            del patcher
        gc.collect()
        if mm is not None:
            try:
                mm.cleanup_models()
                mm.soft_empty_cache()
                return
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.synchronize()

    def unload_by_key(self, cache_key: str) -> bool:
        with self._lock:
            entry = self._cache.pop(cache_key, None)
        if entry is None:
            return False
        self._dispose_entry(entry)
        return True

    def unload_all(self) -> int:
        with self._lock:
            entries = list(self._cache.values())
            self._cache.clear()
        for entry in entries:
            self._dispose_entry(entry)
        return len(entries)

    def get_cache_status(self):
        with self._lock:
            return {
                key: {
                    "model_source": value["model_source"],
                    "dtype": value["dtype"],
                    "device_map": value["device_map"],
                    "attn_implementation": value["attn_implementation"],
                    "last_used": value["last_used"],
                }
                for key, value in self._cache.items()
            }


_CACHE_MANAGER = ModelCacheManager()


def get_cache_manager() -> ModelCacheManager:
    return _CACHE_MANAGER
