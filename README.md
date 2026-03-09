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

Model pairs should be organized as subfolders under:

- `<ComfyUI_ROOT>/models/qwenvision/`

Each model folder must:

- End with `-gguf`
- Contain exactly 2 `.gguf` files
- Include 1 `mmproj` file with filename starting `mmproj-`
- Include 1 vision model `.gguf` file (non-`mmproj`)

Example:

```text
models/qwenvision/
  Qwen3VL-2B-Q4_K_M-gguf/
    Qwen3VL-2B-Instruct-Q4_K_M.gguf
    mmproj-Qwen3VL-2B-Instruct-F16.gguf
```

## How Inference Works

1. `QwenVisionLoader` scans `<ComfyUI_ROOT>/models/qwenvision/*-gguf/` and shows folder names in dropdown.
2. For each selected folder, loader resolves the paired model:
   - `model.gguf` (non-`mmproj`)
   - `mmproj-*.gguf`
3. `QwenVisionRun` executes:
   - `llama-mtmd-cli -m <model.gguf> --mmproj <mmproj.gguf> --image <path> -p <prompt> ...`
4. Output text is returned to ComfyUI as `STRING`.

## Example Workflow

Import:

- `examples/workflow_qwenvision_basic.json`

After import:

1. Set image in `QwenVisionLoadImageWithPath`.
2. Create a `*-gguf` folder under `<ComfyUI_ROOT>/models/qwenvision/` and place paired model files inside.
3. Select that folder name in `QwenVisionLoader`.
4. Queue prompt.

## Notes

- This project does not ship model weights.
- Follow upstream model license and usage terms from Hugging Face.
- This project is community-maintained and not officially affiliated with ComfyUI or Qwen.
