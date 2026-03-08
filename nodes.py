from __future__ import annotations

import traceback

from .qwenvision.cache_manager import get_cache_manager
from .qwenvision.inference import generate_text
from .qwenvision.image_utils import tensor_to_pil_images


class QwenVisionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "Qwen/Qwen2.5-VL-3B-Instruct"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["auto", "float16", "bfloat16", "float32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("QWEN_VISION_MODEL", "STRING")
    RETURN_NAMES = ("qwen_model", "status")
    FUNCTION = "load_model"
    CATEGORY = "Qwen/Vision"

    def load_model(self, model_id: str, device: str, dtype: str):
        manager = get_cache_manager()
        try:
            handle = manager.get_or_load_model(model_id=model_id, device=device, dtype=dtype)
            status = f"Loaded: {handle.model_id} | device={handle.device} | dtype={handle.dtype}"
            return (handle, status)
        except Exception as e:
            err = f"Failed to load model: {e}"
            traceback.print_exc()
            return (None, err)


class QwenVisionRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_VISION_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 4096}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
                "do_sample": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "debug_info")
    FUNCTION = "run"
    CATEGORY = "Qwen/Vision"

    def run(
        self,
        qwen_model,
        image,
        prompt: str,
        max_new_tokens: int,
        system_prompt: str = "",
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: bool = False,
    ):
        if qwen_model is None:
            return ("", "qwen_model is None. Load a model first.")

        try:
            pil_images = tensor_to_pil_images(image)
            if not pil_images:
                return ("", "No valid image found in IMAGE input.")

            # First version: only use the first image
            image_pil = pil_images[0]

            text, debug_info = generate_text(
                handle=qwen_model,
                image_pil=image_pil,
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            return (text, debug_info)
        except Exception as e:
            traceback.print_exc()
            return ("", f"Run failed: {e}")


class QwenVisionUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload_all": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "qwen_model": ("QWEN_VISION_MODEL",),
            }
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
