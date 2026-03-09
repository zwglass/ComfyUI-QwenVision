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


def _scan_local_model_pairs() -> Dict[str, dict]:
    models_dir = _resolve_qwenvision_dir()
    if not os.path.isdir(models_dir):
        return {}

    pairs: Dict[str, dict] = {}
    for folder_name in sorted(os.listdir(models_dir)):
        folder_path = os.path.join(models_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if not folder_name.lower().endswith("-gguf"):
            continue

        gguf_files = []
        for name in sorted(os.listdir(folder_path)):
            full_path = os.path.join(folder_path, name)
            if os.path.isfile(full_path) and name.lower().endswith(".gguf"):
                gguf_files.append(full_path)

        if len(gguf_files) != 2:
            continue

        mmproj_files = [
            path
            for path in gguf_files
            if os.path.basename(path).lower().startswith("mmproj-")
        ]
        if len(mmproj_files) != 1:
            continue

        model_files = [path for path in gguf_files if path not in mmproj_files]
        if len(model_files) != 1:
            continue

        pairs[folder_name] = {
            "model_path": model_files[0],
            "mmproj_path": mmproj_files[0],
        }

    return pairs


class QwenVisionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_pairs = _scan_local_model_pairs()
        folder_names = list(model_pairs.keys())
        default_name = folder_names[0] if folder_names else ""
        return {
            "required": {
                "model_source": (
                    folder_names,
                    {"default": default_name, "label": "QwenVL模型"},
                ),
                "cli_path": ("STRING", {"default": "llama-mtmd-cli"}),
            }
        }

    RETURN_TYPES = ("QWEN_VISION_MODEL", "STRING")
    RETURN_NAMES = ("qwen_model", "status")
    FUNCTION = "load_model"
    CATEGORY = "Qwen/Vision"

    def load_model(self, model_source: str, cli_path: str):
        model_pairs = _scan_local_model_pairs()
        if model_source not in model_pairs:
            return (
                None,
                "Invalid model_source. Put paired models under models/qwenvision/<name>-gguf/.",
            )
        pair = model_pairs[model_source]

        manager = get_cache_manager()
        try:
            handle = manager.get_or_load_model(
                model_source=pair["model_path"],
                mmproj_source=pair["mmproj_path"],
                cli_path=cli_path.strip() or "llama-mtmd-cli",
            )
            status = (
                "Loaded GGUF: "
                f"model={handle.model_path} | mmproj={handle.mmproj_path} | "
                f"cli={handle.cli_path}"
            )
            return (handle, status)
        except Exception as e:
            err = f"Failed to load model files: {e}"
            traceback.print_exc()
            return (None, err)
