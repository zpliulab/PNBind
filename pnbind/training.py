"""Training utilities for PNBind models on precomputed residue graphs."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from pnbind.checkpoints import MODEL_FAMILIES
from pnbind.data import load_feature_graph


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GraphManifestDataset(torch.utils.data.Dataset):
    """Dataset backed by JSONL records with ``graph`` and ``esm3_layers`` paths."""

    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.records = []
        for line_number, raw in enumerate(self.manifest.read_text().splitlines(), start=1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            if "graph" not in record or "esm3_layers" not in record:
                raise ValueError(f"{self.manifest}:{line_number}: requires graph and esm3_layers")
            self.records.append(record)
        if not self.records:
            raise ValueError(f"{self.manifest}: no records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        graph = self.root / record["graph"]
        layers = self.root / record["esm3_layers"]
        data = load_feature_graph(graph, layers)
        if not hasattr(data, "y") or data.y is None:
            raise ValueError(f"{graph}: training graph requires residue labels in y")
        return data


@dataclass
class EpochMetrics:
    loss: float
    residues: int
    f1: float
    mcc: float


def _node_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    return output["node_logits"] if isinstance(output, dict) else output


class FocalLoss(torch.nn.Module):
    """Binary focal loss used for the reported PNBind training procedure."""

    def __init__(self, alpha: float = 0.30, gamma: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        probability = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probability, 1 - probability)
        alpha = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        return (alpha * (1 - pt).pow(self.gamma) * bce).mean()


def _binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    predicted = logits >= 0
    labels = labels.bool()
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    tn = int((~predicted & ~labels).sum())
    denominator = 2 * tp + fp + fn
    f1 = 0.0 if denominator == 0 else 2 * tp / denominator
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = 0.0 if mcc_denominator == 0 else (tp * tn - fp * fn) / mcc_denominator
    return f1, mcc


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss, total_residues = 0.0, 0
    all_logits, all_labels = [], []
    criterion = FocalLoss()
    for data in loader:
        data = data.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = _node_logits(model(data)).reshape(-1)
            labels = data.y.reshape(-1)
            if logits.shape != labels.shape:
                raise RuntimeError(f"logit/label shape mismatch: {logits.shape} vs {labels.shape}")
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        count = int(labels.numel())
        total_loss += float(loss.detach()) * count
        total_residues += count
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    f1, mcc = _binary_metrics(torch.cat(all_logits), torch.cat(all_labels))
    return EpochMetrics(loss=total_loss / total_residues, residues=total_residues, f1=f1, mcc=mcc)


def train(
    config: dict,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    output_dir: str | Path,
) -> Path:
    """Train one PNBind model and return the best validation checkpoint path."""
    required = {"model_family", "model_config", "optimizer", "training"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"training config missing {sorted(missing)}")
    family = config["model_family"]
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model_family {family!r}")

    if "seed" in config:
        seed_everything(int(config["seed"]))
    training_cfg = config["training"]
    if int(training_cfg.get("batch_size", 1)) != 1:
        raise ValueError("batch_size must be 1 because ESM layer tensors are chain-shaped")
    device = torch.device(training_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(GraphManifestDataset(train_manifest), batch_size=1, shuffle=True)
    validation_loader = DataLoader(GraphManifestDataset(validation_manifest), batch_size=1, shuffle=False)

    model = MODEL_FAMILIES[family](**config["model_config"]).to(device)
    opt_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    best_f1, best_mcc, stale_epochs = -1.0, -1.0, 0
    history = []
    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer,
        )
        validation_metrics = run_epoch(
            model, validation_loader, device,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "validation_loss": validation_metrics.loss,
            "validation_f1": validation_metrics.f1,
            "validation_mcc": validation_metrics.mcc,
        })
        # Validation F1 is primary; MCC resolves exact F1 ties.
        improved = (validation_metrics.f1 > best_f1) or (
            validation_metrics.f1 == best_f1 and validation_metrics.mcc > best_mcc
        )
        if improved:
            best_f1, best_mcc, stale_epochs = validation_metrics.f1, validation_metrics.mcc, 0
            torch.save({
                "format_version": 1,
                "model_family": family,
                "model_config": config["model_config"],
                "state_dict": model.state_dict(),
                "training_metadata": {
                    "epoch": epoch,
                    "validation_f1": best_f1,
                    "validation_mcc": best_mcc,
                },
            }, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= int(training_cfg["early_stopping_patience"]):
                break
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return best_path
