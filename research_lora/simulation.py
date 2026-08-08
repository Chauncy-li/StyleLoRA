"""Non-invasive bridge for an already constructed baseline NuPlan simulation planner."""

from __future__ import annotations

from research_lora.model.checkpoint import load_adapter_checkpoint
from research_lora.model.style_lora_planner import StyleLoRAPlanner


def attach_lora_to_simulation_planner(simulation_planner, aggressive_adapter: str, conservative_adapter: str, *, baseline_checkpoint: str,
                                      normalization_file: str, rank: int = 4, rho: float = 0.0) -> StyleLoRAPlanner:
    """Replace only the planner instance's in-memory model after its normal baseline loading.

    The caller remains responsible for constructing the existing baseline planner,
    so NuPlan configuration, feature processing, SDE and DPM-Solver are unchanged.
    """
    if not hasattr(simulation_planner, "_planner"):
        raise TypeError("Expected a constructed baseline simulation planner with a `_planner` model")
    wrapped = StyleLoRAPlanner(simulation_planner._planner, rank=rank)
    load_adapter_checkpoint(aggressive_adapter, wrapped, baseline_checkpoint=baseline_checkpoint,
                            normalization_file=normalization_file)
    load_adapter_checkpoint(conservative_adapter, wrapped, baseline_checkpoint=baseline_checkpoint,
                            normalization_file=normalization_file)
    wrapped.set_strength(rho).eval()
    simulation_planner._planner = wrapped
    return wrapped


class LoRADiffusionPlanner:
    """Lazy proxy that avoids importing NuPlan modules until closed-loop execution.

    ``initialize`` intentionally attaches adapters *after* the parent planner
    loads its baseline checkpoint, so NuPlan reinitialisation cannot overwrite
    the LoRA wrapper.
    """
    def __new__(cls, *args, **kwargs):
        from baseline.simulation.planner import DiffusionPlanner

        class _AttachedLoRADiffusionPlanner(DiffusionPlanner):
            def initialize(self, initialization):
                super().initialize(initialization)
                attach_lora_to_simulation_planner(
                    self,
                    self._lora_aggressive_adapter,
                    self._lora_conservative_adapter,
                    baseline_checkpoint=self._ckpt_path,
                    normalization_file=self._lora_normalization_file,
                    rank=self._lora_rank,
                    rho=self._lora_rho,
                )

            def __init__(self, config, *planner_args, **planner_kwargs):
                self._lora_aggressive_adapter = str(getattr(config, "lora_aggressive_adapter"))
                self._lora_conservative_adapter = str(getattr(config, "lora_conservative_adapter"))
                self._lora_normalization_file = str(getattr(config, "lora_normalization_file", config.normalization_file_path))
                self._lora_rank = int(getattr(config, "lora_rank", 4))
                self._lora_rho = float(getattr(config, "lora_rho", 0.0))
                super().__init__(config, *planner_args, **planner_kwargs)

        return _AttachedLoRADiffusionPlanner(*args, **kwargs)
