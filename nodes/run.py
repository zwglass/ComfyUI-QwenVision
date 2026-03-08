from __future__ import annotations

import traceback

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
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "top_k": ("INT", {"default": 20, "min": 1, "max": 200}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
                "ctx_size": ("INT", {"default": 8192, "min": 1024, "max": 262144}),
                "batch_size": ("INT", {"default": 2048, "min": 32, "max": 8192}),
                "ubatch_size": ("INT", {"default": 512, "min": 32, "max": 4096}),
                "no_warmup": ("BOOLEAN", {"default": True}),
                "timeout_sec": ("INT", {"default": 600, "min": 1, "max": 7200}),
            },
            "optional": {
                "image_path": ("STRING", {"default": ""}),
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
        temperature: float,
        top_k: int,
        top_p: float,
        ctx_size: int,
        batch_size: int,
        ubatch_size: int,
        no_warmup: bool,
        timeout_sec: int,
        image_path: str = "",
    ):
        if qwen_model is None:
            return ("", "qwen_model is None. Load a model first.")

        try:
            text, debug_info = generate_text(
                handle=qwen_model,
                image_tensor=image,
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                ctx_size=ctx_size,
                batch_size=batch_size,
                ubatch_size=ubatch_size,
                no_warmup=no_warmup,
                timeout_sec=timeout_sec,
            )
            return (text, debug_info)
        except Exception as e:
            traceback.print_exc()
            return ("", f"Run failed: {e}")
