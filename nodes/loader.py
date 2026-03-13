from __future__ import annotations

import os
import traceback
from typing import Dict

from ..qwenvision.cache_manager import get_cache_manager


def _resolve_qwenvision_dir() -> str:
    try:
        import folder_paths

        return os.path.join(folder_paths.models_dir, "qwenvision")
    except Exception:
        plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        comfyui_root = os.path.abspath(os.path.join(plugin_root, "..", ".."))
        return os.path.join(comfyui_root, "models", "qwenvision")


def _looks_like_transformers_model_dir(folder_path: str) -> bool:
    if not os.path.isdir(folder_path):
        return False

    names = set(os.listdir(folder_path))
    marker_files = {
        "config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    }
    has_marker = any(name in names for name in marker_files)
    has_weights = any(
        name.endswith((".safetensors", ".bin"))
        or name.startswith("model-")
        for name in names
    )
    return has_marker and has_weights


def _scan_local_transformers_models() -> Dict[str, str]:
    models_dir = _resolve_qwenvision_dir()
    if not os.path.isdir(models_dir):
        return {}

    model_dirs: Dict[str, str] = {}
    for folder_name in sorted(os.listdir(models_dir)):
        folder_path = os.path.join(models_dir, folder_name)
        if not _looks_like_transformers_model_dir(folder_path):
            continue
        model_dirs[folder_name] = folder_path
    return model_dirs


class QwenVisionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_dirs = _scan_local_transformers_models()
        folder_names = list(model_dirs.keys())
        if not folder_names:
            folder_names = [""]
        default_name = folder_names[0]
        return {
            "required": {
                "model_source": (
                    folder_names,
                    {"default": default_name, "label": "Transformers模型"},
                ),
                "dtype": (
                    ["auto", "float16", "bfloat16", "float32"],
                    {"default": "auto"},
                ),
                "device_map": (
                    "STRING",
                    {"default": "auto", "multiline": False},
                ),
                "attn_implementation": (
                    ["default", "flash_attention_2", "sdpa", "eager"],
                    {"default": "default"},
                ),
            }
        }

    RETURN_TYPES = ("QWEN_VISION_MODEL", "STRING")
    RETURN_NAMES = ("qwen_model", "status")
    FUNCTION = "load_model"
    CATEGORY = "Qwen/Vision"

    def load_model(
        self,
        model_source: str,
        dtype: str,
        device_map: str,
        attn_implementation: str,
    ):
        model_dirs = _scan_local_transformers_models()
        if model_source not in model_dirs:
            return (
                None,
                "Invalid model_source. Put a Transformers model directory under "
                "models/qwenvision/.",
            )

        manager = get_cache_manager()
        try:
            local_model_path = model_dirs[model_source]
            handle = manager.get_or_load_model(
                model_source=local_model_path,
                dtype=dtype,
                device_map=device_map.strip() or "auto",
                attn_implementation=attn_implementation,
            )
            action = "Reused cached" if handle.was_reused else "Loaded new"
            status = (
                f"{action} Transformers model: "
                f"model={model_source} | path={handle.model_source} | "
                f"dtype={handle.dtype} | device_map={handle.device_map} | "
                f"attn_implementation={handle.attn_implementation}"
            )
            return (handle, status)
        except Exception as e:
            err = f"Failed to load Transformers model: {e}"
            traceback.print_exc()
            return (None, err)
