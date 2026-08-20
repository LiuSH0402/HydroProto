"""Command-line dispatcher for HydroProto experiments."""

from __future__ import annotations

import sys
from importlib import import_module


COMMANDS = {
    "predict": "hydroproto.predict",
    "mlp": "hydroproto.esm2_mlp",
    "local": "hydroproto.hydrolocal",
    "proto": "hydroproto.hydroproto",
    "mpc": "hydroproto.hydrompc",
    "final": "hydroproto.final_evaluation",
    "embed": "hydroproto.extract_embeddings",
    "embedding-train": "hydroproto.train",
    "embedding-cv": "hydroproto.cross_validate",
    "mmseqs": "hydroproto.mmseqs_baseline",
}


def _usage() -> str:
    commands = "|".join(COMMANDS)
    return (
        "Usage: hydroproto <command> [arguments]\n"
        f"Commands: {commands}\n"
        "Run 'hydroproto <command> --help' for model-specific options."
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command: {command}\n{_usage()}")
    import_module(COMMANDS[command]).main()


if __name__ == "__main__":
    main()
