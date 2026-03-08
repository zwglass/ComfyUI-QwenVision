from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def tensor_to_pil_images(image_tensor) -> List[Image.Image]:
    # Handles common ComfyUI IMAGE layouts:
    # [H, W, C] or [B, H, W, C]
    arr = image_tensor.detach().cpu().numpy()

    if arr.ndim == 3:
        arr = arr[None, ...]

    if arr.ndim != 4:
        raise ValueError(f"Unsupported IMAGE tensor shape: {arr.shape}")

    images = []
    for item in arr:
        item = _to_uint8(item)
        if item.shape[-1] == 1:
            item = np.repeat(item, 3, axis=-1)
        elif item.shape[-1] == 4:
            item = item[..., :3]
        image = Image.fromarray(item, mode="RGB")
        images.append(image)
    return images
