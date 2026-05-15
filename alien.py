#!/usr/bin/env python3
"""Command dispatcher for the Alien Space of Science pipeline."""

import asyncio
import sys
from collections.abc import Callable


COMMANDS: dict[str, tuple[str, str, str]] = {
    "atomize": (
        "atomization.pipeline",
        "run_pipeline",
        "Run PDF -> blog -> idea -> refined idea atomization.",
    ),
    "cluster": (
        "atomization.stages.cluster_atoms",
        "run_clustering",
        "Cluster refined ideas and name idea atoms.",
    ),
    "make-datasets": (
        "generation.make_datasets",
        "main",
        "Generate coherence datasets from clusters.json.",
    ),
    "make-availability": (
        "generation.make_availability",
        "main",
        "Generate two-tower author availability datasets.",
    ),
    "train-coherence": (
        "generation.train_coherence",
        "main",
        "Train the coherence transformer.",
    ),
    "train-availability": (
        "generation.train_availability",
        "main",
        "Train the two-tower availability model.",
    ),
    "sample": (
        "generation.sample",
        "main",
        "Sample high-coherence, low-availability atom sets.",
    ),
    "reconstruct": (
        "generation.reconstruct_cli",
        "main",
        "Reconstruct selected atom sets into methodology sketches.",
    ),
}


def _usage() -> str:
    lines = [
        "Usage: python alien.py <command> [args...]",
        "",
        "Commands:",
    ]
    width = max(len(name) for name in COMMANDS)
    for name, (_, _, description) in COMMANDS.items():
        lines.append(f"  {name:<{width}}  {description}")
    lines.extend([
        "",
        "Run `python alien.py <command> --help` for command-specific options.",
    ])
    return "\n".join(lines)


def _load_callable(module_name: str, function_name: str) -> Callable:
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        raise SystemExit(2)

    module_name, function_name, _ = COMMANDS[command]
    entrypoint = _load_callable(module_name, function_name)

    sys.argv = [f"alien.py {command}", *sys.argv[2:]]
    result = entrypoint()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


if __name__ == "__main__":
    main()
