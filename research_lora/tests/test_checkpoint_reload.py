from pathlib import Path

import torch

from research_lora.model.checkpoint import load_adapter_checkpoint, save_adapter_checkpoint, sha256_file
from research_lora.model.style_lora_planner import StyleLoRAPlanner
from research_lora.tests.helpers import TinyBaseline, tiny_inputs


def test_adapter_checkpoint_reloads_exactly(tmp_path: Path):
    baseline_file, normalizer_file = tmp_path / "base.pt", tmp_path / "norm.json"
    torch.save({"placeholder": 1}, baseline_file); normalizer_file.write_text("{}", encoding="utf-8")
    source = StyleLoRAPlanner(TinyBaseline(), target_modules=("linear",))
    layer = next(module for _, module in source.baseline.named_modules() if hasattr(module, "aggressive"))
    with torch.no_grad(): layer.aggressive.lora_B.fill_(0.1)
    source.set_strength(1); expected = source(tiny_inputs())
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(path, source, style="aggressive", baseline_checkpoint=baseline_file, manifest_hash="m",
                            normalization_file=normalizer_file, training_config={},
                            baseline_sha256=sha256_file(baseline_file),
                            normalization_sha256=sha256_file(normalizer_file))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["adapter_state"] and all(".aggressive.lora_" in key for key in payload["adapter_state"])
    target = StyleLoRAPlanner(TinyBaseline(), target_modules=("linear",))
    load_adapter_checkpoint(path, target, baseline_checkpoint=baseline_file, normalization_file=normalizer_file)
    target.set_strength(1)
    # State equality is the reload contract; input outputs need matching random base weights separately.
    assert source.adapter_state_dict("aggressive").keys() == target.adapter_state_dict("aggressive").keys()
    for name, value in source.adapter_state_dict("aggressive").items(): assert torch.equal(value, target.adapter_state_dict("aggressive")[name])
