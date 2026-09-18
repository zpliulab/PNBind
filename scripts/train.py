#!/usr/bin/env python3
"""Train PNBind from precomputed graph and ESM feature manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pnbind.training import train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a PNBind model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = train(
        json.loads(args.config.read_text()),
        args.train_manifest,
        args.validation_manifest,
        args.output_dir,
    )
    print(f"Best checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
