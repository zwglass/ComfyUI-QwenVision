# ComfyUI-QwenVision

ComfyUI custom nodes for local Qwen vision-language inference.

- Local inference with `transformers`
- Reusable model cache (loader/run/unload workflow)
- Simple ComfyUI graph integration for image understanding tasks

## Features

- `QwenVisionLoader`: load and cache model + processor
- `QwenVisionRun`: run single-image inference with prompt
- `QwenVisionUnload`: unload one model or clear all cached models

## Installation

### 1. Install into ComfyUI custom nodes

```bash
cd <ComfyUI_ROOT>/custom_nodes
git clone https://github.com/zwglass/ComfyUI-QwenVision.git
```

### 2. Install Python dependencies (recommended: uv)

```bash
cd <ComfyUI_ROOT>/custom_nodes/ComfyUI-QwenVision
uv venv
uv sync
```

If you prefer manual install:

```bash
uv run pip install -r requirements.txt
```

### 3. Restart ComfyUI

After restart, search nodes under category `Qwen/Vision`.

## Model Download And Storage Path

Models are loaded from Hugging Face Hub by `transformers`.

- Local model scan path for this node:
  - `<ComfyUI_ROOT>/models/qwenvision/*`
  - Only direct subfolders are listed as local model options.
- Built-in downloadable model option:
  - `Qwen/Qwen3.5-0.8B`

- Default cache path:
  - `~/.cache/huggingface/hub`
- You can change cache location via environment variables:
  - `HF_HOME`
  - `HUGGINGFACE_HUB_CACHE`
  - `TRANSFORMERS_CACHE`

Example (Linux/macOS):

```bash
export HF_HOME=/data/hf_cache
```

Then restart ComfyUI and load model again.

## Manual Downloaded Models (Local Folder Convention)

Manual local models are supported.

- Save each model under:
  - `<ComfyUI_ROOT>/models/qwenvision/<model_folder>`
- Example:
  - `<ComfyUI_ROOT>/models/qwenvision/Qwen3.5-0.8B`
- In `QwenVisionLoader`, pick the model from dropdown.
- Manual string/path input is intentionally disabled.

Required files are typically:

- `config.json`
- tokenizer/processor files (such as `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`)
- model weights (such as `model.safetensors` or sharded `*.safetensors`)

If files are incomplete, loading will fail in `from_pretrained(...)`.

## Supported Models

This plugin currently supports models that can be loaded by:

- `AutoProcessor.from_pretrained(...)`
- `AutoModelForImageTextToText.from_pretrained(...)`
- or fallback `AutoModelForCausalLM.from_pretrained(...)`

Common choices:

- Built-in remote model:
  - `Qwen/Qwen3.5-0.8B` (auto-download when selected)
- Local models:
  - Any model directory under `<ComfyUI_ROOT>/models/qwenvision/` that is compatible with the APIs above

Notes:

- `trust_remote_code=True` is enabled during model load.
- Different model revisions may require newer `transformers`.

## Quick Workflow

1. Add `QwenVisionLoader` and choose `model_id/device/dtype`
2. Connect output `qwen_model` into `QwenVisionRun`
3. Provide `IMAGE` + `prompt`
4. Read `text` output
5. Use `QwenVisionUnload` to release memory when needed

## Project Structure

- `__init__.py`: ComfyUI entry export
- `nodes/`: node definitions (`loader.py`, `run.py`, `unload.py`)
- `qwenvision/cache_manager.py`: global model cache
- `qwenvision/inference.py`: inference pipeline
- `qwenvision/image_utils.py`: IMAGE -> PIL conversion
- `DEVELOPMENT.md`: development rules for openclaw

## GitHub Statements

### Disclaimer

- This project is community-maintained and is not officially affiliated with ComfyUI, Alibaba Cloud, or the Qwen team.
- No model weights are distributed in this repository.

### Model License Notice

- You must follow each model's upstream license and usage policy on Hugging Face.
- Commercial usage depends on the specific model license you choose.

### Security Notice

- This project loads remote model code with `trust_remote_code=True`.
- Only use trusted model repositories.

## Contributing

Issues and PRs are welcome. Please follow `DEVELOPMENT.md`.

## License

No license file is included yet. If you plan to publish publicly, add a license (for example MIT) before release.
