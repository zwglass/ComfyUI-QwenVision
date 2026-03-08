# ComfyUI Qwen Vision Node Development Outline

## Goal

Develop a ComfyUI custom node that performs **image understanding
inference using small Qwen multimodal models** (e.g., Qwen3.5‑0.8B).\
The node should accept an image and prompt, run the model locally using
Transformers, and return text output.

Core stack:

-   Transformers
-   AutoProcessor
-   PyTorch
-   Local singleton model cache

Pipeline:

ComfyUI IMAGE → PIL → Processor → Model.generate() → Decode → TEXT

------------------------------------------------------------------------

# 1. Node Functional Scope

First version should support:

-   Single image input
-   Single prompt
-   Single inference result
-   Local model loading
-   Global model cache

Avoid adding early complexity such as:

-   Batch image processing
-   Video input
-   Structured JSON output
-   Streaming tokens
-   Multi‑round memory
-   Complex model pool switching

------------------------------------------------------------------------

# 2. Node Architecture

Recommended structure: **two nodes**.

### Model Loader Node

    QwenVisionLoader
      ↓
    QWEN_VISION_MODEL

### Inference Node

    IMAGE + PROMPT + QWEN_VISION_MODEL
            ↓
    QwenVisionRun
            ↓
    TEXT

Advantages:

-   Model reuse
-   Better VRAM control
-   Clear separation of loading and inference

------------------------------------------------------------------------

# 3. Project Directory Structure

    ComfyUI-QwenVision/
    ├─ __init__.py
    ├─ nodes.py
    ├─ requirements.txt
    ├─ README.md
    ├─ qwenvision/
    │  ├─ __init__.py
    │  ├─ cache_manager.py
    │  ├─ model_loader.py
    │  ├─ inference.py
    │  ├─ image_utils.py
    │  ├─ prompt_utils.py
    │  ├─ types.py
    │  └─ constants.py
    └─ examples/
       └─ workflow.json

Responsibilities:

  File               Purpose
  ------------------ ------------------------------
  nodes.py           ComfyUI node definitions
  cache_manager.py   global singleton model cache
  model_loader.py    model + processor loading
  inference.py       main inference pipeline
  image_utils.py     tensor → PIL conversion
  prompt_utils.py    message construction
  types.py           internal data types
  constants.py       defaults and configuration

------------------------------------------------------------------------

# 4. Node Parameters

## Loader Node

Parameters:

-   model_id
-   device
-   dtype

Output:

    QWEN_VISION_MODEL

------------------------------------------------------------------------

## Inference Node

Inputs:

-   image
-   prompt
-   system_prompt
-   max_new_tokens
-   temperature
-   top_p
-   do_sample

Outputs:

    text

Optional outputs:

-   raw_text
-   debug_info

------------------------------------------------------------------------

# 5. Model Loading Strategy

Use:

-   AutoProcessor
-   AutoModelForImageTextToText (or compatible model class)

Example:

``` python
processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

------------------------------------------------------------------------

# 6. Input Processing

Images must be converted before inference.

ComfyUI IMAGE → PIL conversion steps:

1.  Extract batch image
2.  Convert float → uint8
3.  Ensure RGB format
4.  Return PIL.Image

------------------------------------------------------------------------

# 7. Message Format

Use multimodal chat messages.

Example:

``` python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": prompt}
        ]
    }
]
```

Then pass messages to processor.

------------------------------------------------------------------------

# 8. Inference Flow

    1. Load model handle
    2. Convert IMAGE → PIL
    3. Build messages
    4. processor(messages)
    5. Move tensors to device
    6. model.generate()
    7. Remove prompt tokens
    8. Decode output
    9. Clean text

------------------------------------------------------------------------

# 9. Singleton Model Cache

Purpose:

-   Prevent repeated model loading
-   Improve inference speed
-   Control GPU memory

Cache key example:

    model_id + device + dtype

Stored data:

    {
      "model": model,
      "processor": processor,
      "model_id": ...,
      "device": ...,
      "dtype": ...,
      "last_used": ...
    }

------------------------------------------------------------------------

# 10. Cache Manager Functions

Required methods:

### get_or_load_model()

Load model if not cached.

### unload_model()

Unload specific model or all models.

### clear_cuda_cache()

    gc.collect()
    torch.cuda.empty_cache()

### get_cache_status()

Return information about cached models.

------------------------------------------------------------------------

# 11. Model Handle Object

Instead of passing raw model objects between nodes, wrap them:

``` python
class QwenVisionModelHandle:

    def __init__(self, model, processor, model_id, device, dtype):

        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
```

Benefits:

-   cleaner node interface
-   metadata tracking
-   easier debugging

------------------------------------------------------------------------

# 12. Error Handling

Important exceptions to handle:

### Model loading failure

-   invalid model path
-   missing dependencies

### CUDA out of memory

Handle:

    torch.cuda.OutOfMemoryError

Possible recovery:

-   clear cache
-   suggest smaller model

### Image format errors

-   empty input
-   invalid dimensions

------------------------------------------------------------------------

# 13. Logging

Recommended debug logs:

-   model_id
-   device
-   dtype
-   input image size
-   prompt length
-   generation parameters
-   cache hit/miss
-   inference time

------------------------------------------------------------------------

# 14. Resource Release Node

Optional node:

    QwenVisionUnload

Purpose:

-   remove model from cache
-   free GPU memory

Implementation:

    del model
    gc.collect()
    torch.cuda.empty_cache()

------------------------------------------------------------------------

# 15. Requirements

Minimal dependencies:

    torch
    transformers
    accelerate
    Pillow
    sentencepiece
    safetensors

Optional (if required):

    einops
    qwen-vl-utils

------------------------------------------------------------------------

# 16. Runtime Validation

Check runtime environment at startup:

-   torch version
-   transformers version
-   CUDA availability
-   GPU name

------------------------------------------------------------------------

# 17. Development Phases

### Phase 1 --- Minimal working node

-   single image
-   fixed model
-   text output

### Phase 2 --- Engineering improvements

-   loader node
-   cache manager
-   logging
-   error handling

### Phase 3 --- Usability

-   system prompt
-   generation parameters
-   unload node

### Phase 4 --- Advanced features

-   multiple models
-   batch images
-   structured output
-   streaming tokens

------------------------------------------------------------------------

# 18. Minimal Implementation Files

First working version only requires:

    __init__.py
    nodes.py
    qwenvision/cache_manager.py
    qwenvision/inference.py

------------------------------------------------------------------------

# 19. Recommended Execution Pipeline

    ComfyUI IMAGE
          ↓
    tensor_to_pil
          ↓
    processor(messages)
          ↓
    cached_model.generate()
          ↓
    decode + clean
          ↓
    TEXT
