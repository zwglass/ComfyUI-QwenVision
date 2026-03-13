from __future__ import annotations

import os
import tempfile
import time
from typing import Tuple

import torch

from .cache_manager import get_cache_manager
from .image_utils import tensor_to_pil_images

try:
    import comfy.model_management as mm
except Exception:
    mm = None


def _resolve_image_path(image_tensor, image_path: str = ""):
    if image_path and os.path.isfile(image_path):
        return image_path, None

    pil_images = tensor_to_pil_images(image_tensor)
    if not pil_images:
        raise ValueError("No valid image found in IMAGE input.")

    fd, temp_path = tempfile.mkstemp(prefix="qwenvision_", suffix=".png")
    os.close(fd)
    pil_images[0].save(temp_path, format="PNG")
    return temp_path, temp_path


def _get_first_image(image_tensor, image_path: str = ""):
    if image_path and os.path.isfile(image_path):
        from PIL import Image

        with Image.open(image_path) as image:
            return image.convert("RGB")

    pil_images = tensor_to_pil_images(image_tensor)
    if not pil_images:
        raise ValueError("No valid image found in IMAGE input.")
    return pil_images[0]


def _move_inputs_to_device(device, inputs):
    if device is None:
        return inputs

    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def generate_text(
    handle,
    image_tensor,
    prompt: str,
    image_path: str = "",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.8,
    timeout_sec: int = 600,
) -> Tuple[str, str]:
    temp_file_to_cleanup = None
    t0 = time.perf_counter()
    manager = get_cache_manager()

    try:
        model, processor, patcher, metadata = manager.get_runtime_by_key(handle.cache_key)
        if patcher is not None and mm is not None:
            mm.load_model_gpu(patcher)
        resolved_image_path, temp_file_to_cleanup = _resolve_image_path(
            image_tensor=image_tensor,
            image_path=image_path,
        )
        image = _get_first_image(image_tensor=image_tensor, image_path=image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt.strip()},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inference_device = getattr(patcher, "load_device", None)
        if inference_device is None:
            try:
                inference_device = next(model.parameters()).device
            except StopIteration:
                inference_device = None
        inputs = _move_inputs_to_device(inference_device, inputs)

        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_k"] = top_k

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = output_text[0].strip() if output_text else ""
        elapsed = time.perf_counter() - t0
        debug_info = (
            f"model={metadata['model_source']} | dtype={metadata['dtype']} | "
            f"device_map={metadata['device_map']} | "
            f"attn_implementation={metadata['attn_implementation']} | "
            f"image={resolved_image_path} | timeout_sec={timeout_sec} | "
            f"elapsed={elapsed:.2f}s"
        )
        return text, debug_info
    finally:
        if temp_file_to_cleanup and os.path.isfile(temp_file_to_cleanup):
            try:
                os.remove(temp_file_to_cleanup)
            except OSError:
                pass
