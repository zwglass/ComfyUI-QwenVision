# ComfyUI-QwenVision

ComfyUI custom nodes for Qwen3-VL inference through `transformers`.

## Features

- Transformers-based model loader with in-process cache
- Model stays loaded in the ComfyUI Python process across runs
- Image loader node with explicit `image_path` output
- Automatic fallback: if `image_path` is missing, node saves a temp image

## Included Nodes

- `QwenVisionLoadImageWithPath`
- `QwenVisionLoader`
- `QwenVisionRun`
- `QwenVisionUnload`

## Installation

1. Clone into ComfyUI custom nodes:

```bash
cd <ComfyUI_ROOT>/custom_nodes
git clone https://github.com/zwglass/ComfyUI-QwenVision.git
```

2. Install dependencies:

```bash
cd <ComfyUI_ROOT>/custom_nodes/ComfyUI-QwenVision
uv venv
uv sync
```

3. Ensure your environment can load the model with `transformers`:

```bash
python -c "from transformers import AutoProcessor"
```

## How Inference Works

1. `QwenVisionLoader` loads a Hugging Face model ID or local Transformers model path.
2. The loaded `model`, `processor`, and Comfy `ModelPatcher` stay cached in the ComfyUI process.
3. `QwenVisionRun` asks ComfyUI to load the patcher onto the active device, then calls `model.generate(...)`.
4. `QwenVisionUnload` releases the cached runtime through ComfyUI's model management path.

## Loader Inputs

- `model_source`: Hugging Face model ID or local model path
- `dtype`: `auto`, `float16`, `bfloat16`, `float32`
- `device_map`: target device for ComfyUI model management, not Hugging Face `accelerate` sharding
- `attn_implementation`: `default`, `flash_attention_2`, `sdpa`, `eager`

`device_map` examples:

- `auto`: let ComfyUI choose the current torch device
- `cuda:0`: pin the model to a specific CUDA device when loaded for inference
- `cpu`: keep inference on CPU

The node no longer passes `device_map` through to Transformers `from_pretrained(..., device_map=...)`. Models are wrapped with ComfyUI's `ModelPatcher` so the built-in "clear VRAM" flow can unload them correctly.

Default example:

```text
Qwen/Qwen3-VL-30B-A3B-Instruct
```

## Run Inputs

- `prompt`
- `max_new_tokens`
- `temperature`
- `top_k`
- `top_p`
- `image` / optional `image_path`

## Notes

- This project no longer uses `llama-mtmd-cli`.
- Large multimodal models may require substantial VRAM.
- `flash_attention_2` depends on your Torch/CUDA environment.
- This project does not ship model weights.
