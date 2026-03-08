from .loader import QwenVisionLoader
from .load_image_with_path import QwenVisionLoadImageWithPath
from .run import QwenVisionRun
from .unload import QwenVisionUnload

NODE_CLASS_MAPPINGS = {
    "QwenVisionLoadImageWithPath": QwenVisionLoadImageWithPath,
    "QwenVisionLoader": QwenVisionLoader,
    "QwenVisionRun": QwenVisionRun,
    "QwenVisionUnload": QwenVisionUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVisionLoadImageWithPath": "Qwen Vision Load Image (With Path)",
    "QwenVisionLoader": "Qwen Vision Loader",
    "QwenVisionRun": "Qwen Vision Run",
    "QwenVisionUnload": "Qwen Vision Unload",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
