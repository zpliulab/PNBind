from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    dataset: str
    n_chains: int
    n_residues: int
    threshold: float
    mcc: float
    f1: float
    recall: float
    precision: float
    specificity: float
    auc: float
    ap: float
    aligned_chains: int

    def as_row(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "n_chains": self.n_chains,
            "n_residues": self.n_residues,
            "threshold": self.threshold,
            "mcc": self.mcc,
            "f1": self.f1,
            "recall": self.recall,
            "precision": self.precision,
            "specificity": self.specificity,
            "auc": self.auc,
            "ap": self.ap,
            "aligned_chains": self.aligned_chains,
        }


def load_three_line_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Parse >chain / sequence / binary-label benchmark records."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError(f"{path}: expected records of exactly three non-empty lines")
    labels = {}
    for offset in range(0, len(lines), 3):
        header, sequence, binary = lines[offset : offset + 3]
        if not header.startswith(">"):
            raise ValueError(f"{path}:{offset + 1}: expected FASTA header")
        chain_id = header[1:].split()[0]
        if chain_id in labels:
            raise ValueError(f"{path}: duplicate chain {chain_id}")
        if len(sequence) != len(binary) or set(binary) - {"0", "1"}:
            raise ValueError(f"{path}: invalid labels for {chain_id}")
        labels[chain_id] = np.fromiter((int(value) for value in binary), dtype=np.int8)
    return labels


def load_predictions(path: str | Path) -> tuple[list[str], dict[str, np.ndarray]]:
    archive = np.load(path, allow_pickle=False)
    required = {"pids", "probs_avg", "chain_offsets"}
    if not required.issubset(archive.files):
        raise ValueError(f"{path}: required arrays are {sorted(required)}")
    pids = [str(pid) for pid in archive["pids"]]
    probabilities = archive["probs_avg"].astype(np.float32, copy=False)
    offsets = archive["chain_offsets"].astype(np.int64, copy=False)
    if len(offsets) != len(pids) + 1 or offsets[0] != 0 or offsets[-1] != len(probabilities):
        raise ValueError(f"{path}: inconsistent chain offsets")
    by_chain = {
        pid: probabilities[offsets[index] : offsets[index + 1]]
        for index, pid in enumerate(pids)
    }
    if len(by_chain) != len(pids):
        raise ValueError(f"{path}: duplicate protein IDs")
    return pids, by_chain


def evaluate_dataset(
    name: str,
    data_path: str | Path,
    prediction_path: str | Path,
    threshold: float,
) -> tuple[EvaluationResult, list[dict[str, object]]]:
    labels = load_three_line_dataset(data_path)
    order, probabilities = load_predictions(prediction_path)
    if set(order) != set(labels):
        missing = sorted(set(labels) - set(order))
        extra = sorted(set(order) - set(labels))
        raise ValueError(f"{name}: chain mismatch; missing={missing}, extra={extra}")

    all_labels, all_probabilities, alignment = [], [], []
    for chain_id in order:
        y_true = labels[chain_id]
        y_score = probabilities[chain_id]
        length = min(len(y_true), len(y_score))
        if len(y_true) != len(y_score):
            alignment.append(
                {
                    "dataset": name,
                    "chain_id": chain_id,
                    "label_length": len(y_true),
                    "prediction_length": len(y_score),
                    "used_length": length,
                }
            )
        all_labels.append(y_true[:length])
        all_probabilities.append(y_score[:length])

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probabilities)
    y_pred = y_score >= threshold
    negatives = y_true == 0
    specificity = float(((~y_pred) & negatives).sum() / negatives.sum())
    result = EvaluationResult(
        dataset=name,
        n_chains=len(order),
        n_residues=len(y_true),
        threshold=float(threshold),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred)),
        specificity=specificity,
        auc=float(roc_auc_score(y_true, y_score)),
        ap=float(average_precision_score(y_true, y_score)),
        aligned_chains=len(alignment),
    )
    return result, alignment


def run_config(config_path: str | Path):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = json.loads(config_path.read_text())
    results, alignment = [], []
    for item in config["datasets"]:
        result, changes = evaluate_dataset(
            item["name"],
            root / item["data"],
            root / item["predictions"],
            item["threshold"],
        )
        results.append(result)
        alignment.extend(changes)
    return results, alignment


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
