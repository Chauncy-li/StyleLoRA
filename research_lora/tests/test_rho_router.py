import torch

from research_lora.model.style_lora_planner import StyleLoRAPlanner
from research_lora.tests.helpers import TinyBaseline, tiny_inputs


def test_rho_router_is_mutually_exclusive_and_zero_is_identity():
    model = StyleLoRAPlanner(TinyBaseline(), target_modules=("linear",))
    layer = next(module for _, module in model.baseline.named_modules() if hasattr(module, "aggressive"))
    with torch.no_grad():
        layer.aggressive.lora_A.fill_(1); layer.aggressive.lora_B.fill_(1)
        layer.conservative.lora_A.fill_(2); layer.conservative.lora_B.fill_(1)
    x = tiny_inputs(); base = model.set_strength(0)(x)
    positive = model.set_strength(1)(x); negative = model.set_strength(-1)(x)
    assert torch.equal(base, model.baseline.linear.base(x["x"]))
    assert not torch.equal(positive, negative)
    assert model._style == "cons"
