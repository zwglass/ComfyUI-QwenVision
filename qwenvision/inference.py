from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import Tuple

from .image_utils import tensor_to_pil_images


def _resolve_image_path(image_tensor, image_path: str = ""):
    if image_path and os.path.isfile(image_path):
        return image_path, None

    pil_images = tensor_to_pil_images(image_tensor)
    if not pil_images:
        raise ValueError("No valid image found in IMAGE input.")

    fd, temp_path = tempfile.mkstemp(prefix="qwenvision_", suffix=".png")
    os.close(fd)
    pil_images[0].save(temp_path, format="PNG")
    return temp_path, temp_path


def generate_text(
    handle,
    image_tensor,
    prompt: str,
    image_path: str = "",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.8,
    ctx_size: int = 8192,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    no_warmup: bool = True,
    timeout_sec: int = 600,
) -> Tuple[str, str]:
    temp_file_to_cleanup = None
    t0 = time.perf_counter()

    try:
        resolved_image_path, temp_file_to_cleanup = _resolve_image_path(
            image_tensor=image_tensor,
            image_path=image_path,
        )

        cmd = [
            handle.cli_path,
            "-m",
            handle.model_path,
            "--mmproj",
            handle.mmproj_path,
            "--ctx-size",
            str(ctx_size),
            "--batch-size",
            str(batch_size),
            "--ubatch-size",
            str(ubatch_size),
            "--image",
            resolved_image_path,
            "-p",
            prompt.strip(),
            "--temp",
            str(temperature),
            "--top-k",
            str(top_k),
            "--top-p",
            str(top_p),
            "-n",
            str(max_new_tokens),
        ]
        if no_warmup:
            cmd.append("--no-warmup")

        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_sec),
        )

        if proc.returncode != 0:
            stderr_text = (proc.stderr or "").strip()
            raise RuntimeError(
                f"llama-mtmd-cli failed with code {proc.returncode}: {stderr_text}"
            )

        text = (proc.stdout or "").strip()
        elapsed = time.perf_counter() - t0
        debug_info = (
            f"cli={handle.cli_path} | model={handle.model_path} | "
            f"mmproj={handle.mmproj_path} | image={resolved_image_path} | "
            f"ctx_size={ctx_size} | batch_size={batch_size} | "
            f"ubatch_size={ubatch_size} | no_warmup={no_warmup} | "
            f"elapsed={elapsed:.2f}s"
        )
        return text, debug_info
    finally:
        if temp_file_to_cleanup and os.path.isfile(temp_file_to_cleanup):
            try:
                os.remove(temp_file_to_cleanup)
            except OSError:
                pass
