#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pnbind.checkpoints import checkpoint_paths, ensemble_predict
from pnbind.data import load_feature_graph


EXAMPLES = {
    "DNA": (
        ROOT / "examples/graph_cache/DNA_Test_129_chains/4zm2_B.pt",
        ROOT / "examples/esm3_v3_embeddings/DNA_Test_129_chains/4zm2_B.pt",
        ROOT / "examples/expected_DNA_4zm2_B.npz",
    ),
    "RNA": (
        ROOT / "examples/graph_cache/RNA_Test_117_chains/5o9z_N.pt",
        ROOT / "examples/esm3_v3_embeddings/RNA_Test_117_chains/5o9z_N.pt",
        ROOT / "examples/expected_RNA_5o9z_N.npz",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the eight-checkpoint PNBind ensemble on precomputed features")
    parser.add_argument("--task", choices=("DNA", "RNA"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--esm3-layers", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "results/example_probabilities.csv")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    default_graph, default_layers, expected_path = EXAMPLES[args.task]
    graph = args.graph or default_graph
    layers = args.esm3_layers or default_layers
    data = load_feature_graph(graph, layers)
    paths = checkpoint_paths(args.checkpoint_dir, args.task, ROOT / "checkpoints/manifest.json")
    probabilities = ensemble_predict(data, paths, args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("residue_index", "probability"))
        writer.writerows((index + 1, f"{value:.9f}") for index, value in enumerate(probabilities))
    print(f"Wrote {len(probabilities)} residue probabilities to {args.output}")

    if args.check:
        expected = np.load(expected_path, allow_pickle=False)["probabilities"]
        maximum = float(np.max(np.abs(probabilities - expected)))
        print(f"Maximum absolute difference from the packaged reference: {maximum:.3e}")
        if not np.allclose(probabilities, expected, rtol=1e-5, atol=1e-6):
            print("CHECK FAILED", file=sys.stderr)
            return 1
        print("CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
