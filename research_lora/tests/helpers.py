from __future__ import annotations

import torch
from torch import nn


class TinyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, inputs):
        return self.linear(inputs["x"])


def tiny_inputs():
    return {"x": torch.randn(2, 4, 3)}
