from __future__ import annotations

import os
import hashlib

import numpy as np
import torch
from PIL import Image, ImageOps


class QwenVisionLoadImageWithPath:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            input_images = folder_paths.get_filename_list("input")
        except Exception:
            input_images = []

        return {
            "required": {
                "image": (
                    sorted(input_images),
                    {"image_upload": True},
                ),
            }
        }

    CATEGORY = "Qwen/Vision"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_path")
    FUNCTION = "load_image"

    @classmethod
    def IS_CHANGED(cls, image: str):
        try:
            import folder_paths

            image_path = folder_paths.get_annotated_filepath(image)
            with open(image_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return image

    @classmethod
    def VALIDATE_INPUTS(cls, image: str):
        try:
            import folder_paths

            if not folder_paths.exists_annotated_filepath(image):
                return f"Invalid image file: {image}"
        except Exception:
            if not os.path.isfile(image):
                return f"Invalid image file: {image}"
        return True

    def _resolve_image_path(self, image_name: str) -> str:
        try:
            import folder_paths

            return folder_paths.get_annotated_filepath(image_name)
        except Exception:
            return image_name

    def load_image(self, image: str):
        image_path = self._resolve_image_path(image)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, ...]
        image_name = os.path.basename(image_path)
        ui = {
            "images": [
                {
                    "filename": image_name,
                    "subfolder": "",
                    "type": "input",
                }
            ]
        }
        return {"ui": ui, "result": (tensor, image_path)}
