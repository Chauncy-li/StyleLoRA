import torch
from torch import nn

from research_lora.model.lora_layers import EgoMaskedLoRALinear, LoRALinear


def test_zero_b_initialization_is_exact_identity():
    linear = nn.Linear(3, 2)
    adapter = LoRALinear(linear, rank=2)
    x = torch.randn(2, 3)
    assert torch.equal(adapter(x), linear(x))


def test_zero_strength_is_exact_identity():
    linear = nn.Linear(3, 2)
    adapter = EgoMaskedLoRALinear(linear, rank=2)
    adapter.strength = 0.0
    x = torch.randn(2, 4, 3)
    assert torch.equal(adapter(x), linear(x))
