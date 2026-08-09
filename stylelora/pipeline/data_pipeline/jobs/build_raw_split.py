"""Build raw-DB manifests for cache generation and simulation-only testing.

This command partitions log names but does not process scenarios or generate
``.npz`` files. Run it when establishing a split, changing the raw DB list, or
changing the held-out fraction/seed. Use
``stylelora.scripts.process_cache`` for ordinary cache rebuilds.
"""

from __future__ import annotations

from stylelora.pipeline.data_pipeline.raw_split import main


if __name__ == "__main__":
    main()

