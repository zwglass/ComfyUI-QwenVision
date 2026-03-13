from __future__ import annotations

from ..qwenvision.cache_manager import get_cache_manager


class QwenVisionUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload_all": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "qwen_model": ("QWEN_VISION_MODEL",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "Qwen/Vision"

    def unload(self, unload_all: bool, qwen_model=None):
        manager = get_cache_manager()

        if unload_all:
            unload_count = manager.unload_all()
            return (f"Unloaded {unload_count} cached Transformers model(s).",)

        if qwen_model is None:
            return ("No model handle provided.",)

        unloaded = manager.unload_by_key(qwen_model.cache_key)
        if not unloaded:
            return (f"Model was not cached: {qwen_model.cache_key}",)
        return (f"Unloaded model: {qwen_model.cache_key}",)
