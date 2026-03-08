from __future__ import annotations

import os
import traceback

from ..qwenvision.cache_manager import get_cache_manager

REMOTE_MODEL_OPTIONS = [
    "Qwen/Qwen3.5-0.8B",
]


def _resolve_qwenvision_dir() -> str:
    try:
        import folder_paths

        return os.path.join(folder_paths.models_dir, "qwenvision")
    except Exception:
        plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        comfyui_root = os.path.abspath(os.path.join(plugin_root, "..", ".."))
        return os.path.join(comfyui_root, "models", "qwenvision")


def _scan_local_models() -> list[str]:
    models_dir = _resolve_qwenvision_dir()
    if not os.path.isdir(models_dir):
        return []

    local_models: list[str] = []
    for item in sorted(os.listdir(models_dir)):
        full_path = os.path.join(models_dir, item)
        if os.path.isdir(full_path):
            local_models.append(full_path)
    return local_models


def _model_options() -> list[str]:
    options = _scan_local_models() + REMOTE_MODEL_OPTIONS
    return options if options else REMOTE_MODEL_OPTIONS


class QwenVisionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        options = _model_options()
        return {
            "required": {
                "model_id": (options, {"default": options[0]}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (
                    ["auto", "float16", "bfloat16", "float32"],
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("QWEN_VISION_MODEL", "STRING")
    RETURN_NAMES = ("qwen_model", "status")
    FUNCTION = "load_model"
    CATEGORY = "Qwen/Vision"

    def load_model(self, model_id: str, device: str, dtype: str):
        options = _model_options()
        if model_id not in options:
            return (
                None,
                (
                    "Invalid model_id selection. Choose an existing local model "
                    "under ComfyUI/models/qwenvision or a built-in downloadable "
                    "model."
                ),
            )

        manager = get_cache_manager()
        try:
            handle = manager.get_or_load_model(
                model_id=model_id, device=device, dtype=dtype
            )
            status = (
                f"Loaded: {handle.model_id} | device={handle.device} | "
                f"dtype={handle.dtype}"
            )
            return (handle, status)
        except Exception as e:
            err = f"Failed to load model: {e}"
            traceback.print_exc()
            return (None, err)
