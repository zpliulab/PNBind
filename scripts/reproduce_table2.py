#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pnbind.evaluation import run_config, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the four PNBind rows in Table 2")
    parser.add_argument("--config", type=Path, default=ROOT / "benchmarks/config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced_pnbind.csv")
    parser.add_argument("--alignment-log", type=Path, default=ROOT / "results/alignment_log.csv")
    parser.add_argument("--check", action="store_true", help="compare against the committed four-decimal results")
    args = parser.parse_args()

    results, alignment = run_config(args.config)
    rows = [result.as_row() for result in results]
    write_csv(args.output, rows)
    write_csv(args.alignment_log, alignment)

    print("dataset          n_chains  n_res   thr     MCC      F1  Recall  Precision  Spe     AUC      AP")
    for result in results:
        print(
            f"{result.dataset:16s} {result.n_chains:8d} {result.n_residues:6d} "
            f"{result.threshold:5.2f} {result.mcc:7.4f} {result.f1:7.4f} "
            f"{result.recall:7.4f} {result.precision:10.4f} {result.specificity:6.4f} "
            f"{result.auc:7.4f} {result.ap:7.4f}"
        )
    print(f"\nAlignment log: {args.alignment_log} ({len(alignment)} chains)")

    if args.check:
        with (ROOT / "results/pnbind_expected.csv").open(newline="") as handle:
            expected = {row["dataset"]: row for row in csv.DictReader(handle)}
        fields = ("mcc", "f1", "recall", "precision", "specificity", "auc", "ap")
        failures = []
        for result in results:
            target = expected[result.dataset]
            if result.n_chains != int(target["n_chains"]) or result.n_residues != int(target["n_residues"]):
                failures.append(f"{result.dataset}: count mismatch")
            for field in fields:
                if round(getattr(result, field), 4) != float(target[field]):
                    failures.append(
                        f"{result.dataset} {field}: {getattr(result, field):.8f} != {target[field]}"
                    )
        if failures:
            print("CHECK FAILED", file=sys.stderr)
            print("\n".join(failures), file=sys.stderr)
            return 1
        print("CHECK PASSED: all four PNBind rows match the committed values at four decimals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
