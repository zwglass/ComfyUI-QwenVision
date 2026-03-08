from __future__ import annotations

import traceback

from ..qwenvision.image_utils import tensor_to_pil_images
from ..qwenvision.inference import generate_text


class QwenVisionRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_VISION_MODEL",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe this image."},
                ),
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
