from research_lora.model.injector import assert_only_lora_trainable
from research_lora.model.style_lora_planner import StyleLoRAPlanner
from research_lora.tests.helpers import TinyBaseline


def test_only_lora_parameters_are_trainable():
    model = StyleLoRAPlanner(TinyBaseline(), target_modules=("linear",))
    assert_only_lora_trainable(model)
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
