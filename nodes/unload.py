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
            manager.unload_all()
            return ("Unloaded all cached Qwen vision models.",)

        if qwen_model is None:
            return ("No model handle provided.",)

        manager.unload_by_key(qwen_model.cache_key)
        return (f"Unloaded model: {qwen_model.cache_key}",)
