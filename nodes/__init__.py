from .loader import QwenVisionLoader
from .run import QwenVisionRun
from .unload import QwenVisionUnload

NODE_CLASS_MAPPINGS = {
    "QwenVisionLoader": QwenVisionLoader,
    "QwenVisionRun": QwenVisionRun,
    "QwenVisionUnload": QwenVisionUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVisionLoader": "Qwen Vision Loader",
    "QwenVisionRun": "Qwen Vision Run",
    "QwenVisionUnload": "Qwen Vision Unload",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
