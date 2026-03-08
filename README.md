# ComfyUI-QwenVision

Minimal ComfyUI custom node skeleton for local multimodal inference with:

- Transformers
- AutoProcessor
- local singleton model cache

## Included nodes

- `QwenVisionLoader`
- `QwenVisionRun`
- `QwenVisionUnload`

## Notes

This is a starter skeleton intended for development and debugging.
You will likely need to adjust the model loading class depending on the
exact Qwen vision-capable model you use.

## Basic layout

- `__init__.py`
- `nodes.py`
- `qwenvision/cache_manager.py`
- `qwenvision/inference.py`
- `qwenvision/image_utils.py`
