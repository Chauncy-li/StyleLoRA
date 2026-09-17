# Machine-specific paths

StyleLoRA reads machine-specific paths from `paths.local.json`. This file is
ignored by Git so each contributor can keep their own dataset and output paths
without overwriting anyone else's configuration.

On a new machine, copy `paths.example.json` to `paths.local.json` and edit the
values under `paths`. The example contains the current server defaults. For a
local computer, replace the server-only paths with that computer's dataset,
map, cache, and output locations. Values may refer to other keys using
`${key}`, so changing `record_root` also updates the default source, output,
and cache paths.

Python code can read a path with:

```python
from stylelora.config.runtime_paths import get_path

cache_root = get_path("cache_root")
```

Shell scripts can read a value with:

```bash
python stylelora/config/runtime_paths.py --get cache_root
```

Add any new machine-specific path as another key under `paths`, then have the
script read that key through `get_path()` or `runtime_paths.py --get`. Existing
environment variables remain supported and take precedence where applicable.
To store the config elsewhere, set `STYLELORA_PATHS_FILE` to its location.

Inspect the active resolved configuration with:

```bash
python stylelora/config/runtime_paths.py --show
```
