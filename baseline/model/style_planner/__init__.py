"""Style-planner model package.

Keep lightweight Preference Flow imports independent of the full encoder/training
stack.  The planner class is still exported with the same public name, but is
loaded only when a caller actually requests it.
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from baseline.model.style_planner.diffusion_planner import Diffusion_Planner


def __getattr__(name: str):
    if name == "Diffusion_Planner":
        from baseline.model.style_planner.diffusion_planner import Diffusion_Planner

        return Diffusion_Planner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Diffusion_Planner"]
