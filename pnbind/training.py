"""Training utilities for PNBind models on precomputed residue graphs."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def _node_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    return output["node_logits"] if isinstance(output, dict) else output


def _loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float) -> torch.Tensor:
    weight = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=weight)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss, total_residues = 0.0, 0
    for data in loader:
        data = data.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = _node_logits(model(data)).reshape(-1)
            labels = data.y.reshape(-1)
            if logits.shape != labels.shape:
                raise RuntimeError(f"logit/label shape mismatch: {logits.shape} vs {labels.shape}")
            loss = _loss(logits, labels, pos_weight)
            if training:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        count = int(labels.numel())
        total_loss += float(loss.detach()) * count
        total_residues += count
    return EpochMetrics(loss=total_loss / total_residues, residues=total_residues)


def train(
    config: dict,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    output_dir: str | Path,
) -> Path:
    """Train one PNBind model and return the best validation checkpoint path."""
    required = {"seed", "model_family", "model_config", "optimizer", "training"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"training config missing {sorted(missing)}")
    family = config["model_family"]
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model_family {family!r}")

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_cfg.get("lr_decay_factor", 0.5)),
        patience=int(training_cfg.get("lr_scheduler_patience", 3)),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    best_loss, stale_epochs = float("inf"), 0
    history = []
    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            model, train_loader, device, float(training_cfg["pos_weight"]), optimizer,
            float(training_cfg["grad_clip_norm"]),
        )
        validation_metrics = run_epoch(
            model, validation_loader, device, float(training_cfg["pos_weight"]),
        )
        scheduler.step(validation_metrics.loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "validation_loss": validation_metrics.loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if validation_metrics.loss < best_loss:
            best_loss, stale_epochs = validation_metrics.loss, 0
            torch.save({
                "format_version": 1,
                "model_family": family,
                "model_config": config["model_config"],
                "state_dict": model.state_dict(),
                "training_metadata": {"epoch": epoch, "validation_loss": best_loss, "config": config},
            }, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= int(training_cfg["early_stopping_patience"]):
                break
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return best_path
