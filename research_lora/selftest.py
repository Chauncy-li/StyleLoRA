"""Run all required LoRA invariants locally before launching remote training."""

from __future__ import annotations

import importlib
import inspect
import pkgutil


def main() -> None:
    import research_lora.tests as tests
    failures = []
    for module_info in pkgutil.iter_modules(tests.__path__, tests.__name__ + "."):
        if not module_info.name.rsplit(".", 1)[-1].startswith("test_"):
            continue
        module = importlib.import_module(module_info.name)
        for name in dir(module):
            if not name.startswith("test_"):
                continue
            test_function = getattr(module, name)
            try:
                parameters = inspect.signature(test_function).parameters
                if not parameters:
                    test_function()
                elif tuple(parameters) == ("tmp_path",):
                    from tempfile import TemporaryDirectory
                    from pathlib import Path
                    with TemporaryDirectory() as temporary:
                        test_function(Path(temporary))
                else:
                    raise TypeError(
                        f"selftest supports only no-argument tests or a single tmp_path fixture; "
                        f"{module_info.name}:{name} has parameters {tuple(parameters)}"
                    )
                print(f"PASS {module_info.name}:{name}")
            except Exception as exc:  # collect every invariant failure before returning non-zero
                failures.append(f"{module_info.name}:{name}: {exc}")
                print(f"FAIL {failures[-1]}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
