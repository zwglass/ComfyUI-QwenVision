# ComfyUI-QwenVision

ComfyUI custom nodes for Qwen3-VL GGUF inference through `llama-mtmd-cli`.

## Features

- GGUF + `mmproj` model loader with local cache
- Inference via subprocess (`llama-mtmd-cli`)
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

2. Install dependencies (recommended: `uv`):

```bash
cd <ComfyUI_ROOT>/custom_nodes/ComfyUI-QwenVision
uv venv
uv sync
```

3. Ensure `llama-mtmd-cli` is installed and in `PATH`:

```bash
llama-mtmd-cli --help
```

## Model Download And Storage Path

Default auto-download model source:

- `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf`

Default auto-download mmproj source:

- `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-F16.gguf`

Downloaded files are materialized under:

- `<ComfyUI_ROOT>/models/qwenvision/<repo_name>/...`

Local dropdown scan path:

- `<ComfyUI_ROOT>/models/qwenvision/**/*.gguf`

## How Inference Works

1. `QwenVisionLoader` resolves GGUF/MMProj sources into local files.
2. `QwenVisionRun` executes:
   - `llama-mtmd-cli -m <model.gguf> --mmproj <mmproj.gguf> --image <path> -p <prompt> ...`
3. Output text is returned to ComfyUI as `STRING`.

## Example Workflow

Import:

- `examples/workflow_qwenvision_basic.json`

After import:

1. Set image in `QwenVisionLoadImageWithPath`.
2. Keep default model URL or choose local GGUF files.
3. Queue prompt.

## Notes

- This project does not ship model weights.
- Follow upstream model license and usage terms from Hugging Face.
- This project is community-maintained and not officially affiliated with ComfyUI or Qwen.
