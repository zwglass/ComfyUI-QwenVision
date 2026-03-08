from __future__ import annotations

import os
import traceback

from ..qwenvision.cache_manager import get_cache_manager

DEFAULT_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/"
    "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
)
DEFAULT_MMPROJ_URL = (
    "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/"
    "mmproj-Qwen3VL-2B-Instruct-F16.gguf"
)


def _resolve_qwenvision_dir() -> str:
    try:
        import folder_paths

        return os.path.join(folder_paths.models_dir, "qwenvision")
    except Exception:
        plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        comfyui_root = os.path.abspath(os.path.join(plugin_root, "..", ".."))
        return os.path.join(comfyui_root, "models", "qwenvision")


def _scan_local_gguf_files():
    models_dir = _resolve_qwenvision_dir()
    if not os.path.isdir(models_dir):
        return [], []

    model_files = []
    mmproj_files = []

    for root, _, files in os.walk(models_dir):
        for name in sorted(files):
            if not name.lower().endswith(".gguf"):
                continue
            full_path = os.path.join(root, name)
            if "mmproj" in name.lower():
                mmproj_files.append(full_path)
            else:
                model_files.append(full_path)

    return model_files, mmproj_files


def _model_options() -> list[str]:
    local_models, _ = _scan_local_gguf_files()
    return local_models + [DEFAULT_MODEL_URL]


def _mmproj_options() -> list[str]:
    _, local_mmproj = _scan_local_gguf_files()
    return local_mmproj + [DEFAULT_MMPROJ_URL]


class QwenVisionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_options = _model_options()
        mmproj_options = _mmproj_options()
        return {
            "required": {
                "model_source": (model_options, {"default": model_options[-1]}),
                "mmproj_source": (
                    mmproj_options,
                    {"default": mmproj_options[-1]},
                ),
                "cli_path": ("STRING", {"default": "llama-mtmd-cli"}),
            }
        }

    RETURN_TYPES = ("QWEN_VISION_MODEL", "STRING")
    RETURN_NAMES = ("qwen_model", "status")
    FUNCTION = "load_model"
    CATEGORY = "Qwen/Vision"

    def load_model(self, model_source: str, mmproj_source: str, cli_path: str):
        model_options = _model_options()
        mmproj_options = _mmproj_options()
        if model_source not in model_options:
            return (
                None,
                "Invalid model_source. Choose local .gguf or default download URL.",
            )
        if mmproj_source not in mmproj_options:
            return (
                None,
                "Invalid mmproj_source. Choose local mmproj .gguf or default URL.",
            )

        manager = get_cache_manager()
        try:
            handle = manager.get_or_load_model(
                model_source=model_source,
                mmproj_source=mmproj_source,
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
