from __future__ import annotations

from typing import Any

import torch


class QwenVisionManagedModel(torch.nn.Module):
    def __init__(self, hf_model: torch.nn.Module, initial_device: torch.device) -> None:
        super().__init__()
        self.model = hf_model
        self.device = initial_device

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
