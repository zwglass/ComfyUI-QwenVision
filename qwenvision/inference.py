from __future__ import annotations

import time
from typing import Tuple

import torch


def _build_messages(image_pil, prompt: str, system_prompt: str = ""):
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt.strip()}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": prompt.strip()},
            ],
        }
    )
    return messages


def _move_inputs_to_model_device(inputs, model):
    if hasattr(model, "device"):
        return inputs.to(model.device)
    return inputs


def generate_text(
    handle,
    image_pil,
    prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = False,
) -> Tuple[str, str]:
    model = handle.model
    processor = handle.processor

    messages = _build_messages(
        image_pil=image_pil,
        prompt=prompt,
        system_prompt=system_prompt,
    )

    t0 = time.perf_counter()

    # Preferred path: multimodal chat template support.
    if hasattr(processor, "apply_chat_template"):
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text_prompt],
            images=[image_pil],
            padding=True,
            return_tensors="pt",
        )
    else:
        # Basic fallback path
        inputs = processor(
            text=prompt,
            images=image_pil,
            return_tensors="pt",
        )

    inputs = _move_inputs_to_model_device(inputs, model)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    generated_only = output_ids[:, input_len:] if input_len > 0 else output_ids

    text = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    elapsed = time.perf_counter() - t0
    debug_info = (
        f"model_id={handle.model_id} | device={handle.device} | dtype={handle.dtype} | "
        f"prompt_len={len(prompt)} | image_size={image_pil.size} | elapsed={elapsed:.2f}s"
    )

    return text, debug_info
