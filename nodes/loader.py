from __future__ import annotations

import traceback

from ..qwenvision.cache_manager import get_cache_manager


class QwenVisionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "Qwen/Qwen2.5-VL-3B-Instruct"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (
                    ["auto", "float16", "bfloat16", "float32"],
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("QWEN_VISION_MODEL", "STRING")
    RETURN_NAMES = ("qwen_model", "status")
    FUNCTION = "load_model"
    CATEGORY = "Qwen/Vision"

    def load_model(self, model_id: str, device: str, dtype: str):
        manager = get_cache_manager()
        try:
            handle = manager.get_or_load_model(
                model_id=model_id, device=device, dtype=dtype
            )
            status = (
                f"Loaded: {handle.model_id} | device={handle.device} | "
                f"dtype={handle.dtype}"
            )
            return (handle, status)
        except Exception as e:
            err = f"Failed to load model: {e}"
            traceback.print_exc()
            return (None, err)
