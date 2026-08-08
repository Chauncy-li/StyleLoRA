import torch
from torch import nn

from research_lora.model.lora_layers import EgoMaskedLoRALinear


def test_adapter_writes_only_ego_token():
    layer = EgoMaskedLoRALinear(nn.Linear(3, 2, bias=False), rank=1)
    with torch.no_grad():
        layer.lora_A.fill_(1); layer.lora_B.fill_(1)
    x = torch.randn(2, 5, 3)
    delta = layer(x) - layer.base(x)
    assert torch.count_nonzero(delta[:, 1:]) == 0
    assert torch.count_nonzero(delta[:, 0]) > 0
