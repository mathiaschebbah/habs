# %%
"""Replay the registered synthetic mechanisms on the observed cohort-2 design."""

import argparse
import json
from pathlib import Path

from simulation import run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_file", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-seconds", type=float, default=2400)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--frozen", type=Path)
    arguments = parser.parse_args()
    run(
        arguments.feature_file,
        arguments.out_dir,
        replicates=arguments.replicates,
        workers=arguments.workers,
        max_seconds=arguments.max_seconds,
        resume=arguments.resume,
        expected_feature_sha=(
            json.loads(arguments.frozen.read_text())["feature_sha256"]
            if arguments.frozen
            else None
        ),
    )
