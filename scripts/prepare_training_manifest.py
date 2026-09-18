#!/usr/bin/env python3
"""Create deterministic train/validation JSONL manifests from precomputed features."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def manifest_path(path: Path, root: Path) -> str:
    """Prefer portable relative paths, while accepting external feature stores."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PNBind training manifests")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--esm3-layers-dir", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=5002)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    graph_dir = args.graph_dir.resolve()
    layer_dir = args.esm3_layers_dir.resolve()
    root = Path.cwd().resolve()
    records = []
    for graph in sorted(graph_dir.glob("*.pt")):
        layers = layer_dir / graph.name
        if layers.is_file():
            records.append({
                "graph": manifest_path(graph, root),
                "esm3_layers": manifest_path(layers, root),
            })
    if len(records) < 2:
        raise ValueError("need at least two graph/layer pairs")
    random.Random(args.seed).shuffle(records)
    validation_count = max(1, round(len(records) * args.validation_fraction))
    write_records(args.validation_output, records[:validation_count])
    write_records(args.train_output, records[validation_count:])
    print(f"train={len(records) - validation_count} validation={validation_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
