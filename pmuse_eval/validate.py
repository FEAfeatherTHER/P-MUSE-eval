"""Validate one P-MUSE test-set and submission partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .manifests import BENCHMARKS, TASKS, validate_testset
from .submission import validate_submission


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a P-MUSE submission")
    parser.add_argument("--testset-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--expected-rows", type=int, default=400)
    parser.add_argument("--skip-audio-check", action="store_true")
    args = parser.parse_args(argv)

    payload = {"testset": validate_testset(args.testset_root, args.expected_rows)}
    if args.submission_root is not None:
        payload["submission"] = validate_submission(
            args.testset_root,
            args.submission_root,
            args.benchmark,
            args.task,
            check_audio=not args.skip_audio_check,
        )
        payload["submission"].pop("samples", None)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
