"""Airflow-safe command wrapper for one recommender retraining stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from recommender.orchestration.pipeline import run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--mode",
        choices=("validate", "full"),
        default=os.getenv("RECSYS_DAG_MODE", "validate"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(
        os.getenv("RECSYS_PROJECT_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    outputs = run_stage(args.stage, args.mode, project_root)
    print(f"{args.stage}: validated {len(outputs)} output(s)")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
