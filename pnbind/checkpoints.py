from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from pnbind.models.pnbind_v2_gvp_gate_fusion import PNBindGVPGateFusion
from pnbind.models.pnbind_v2_gvp_hier_esm import PNBindGVPHierESM
from pnbind.models.pnbind_v2_surface_enhanced import PNBindSurfaceEnhanced


MODEL_FAMILIES = {
    "gate_fusion": PNBindGVPGateFusion,
    "hier_esm": PNBindGVPHierESM,
    "surface_enhanced": PNBindSurfaceEnhanced,
}


def load_checkpoint(path: str | Path, device: str | torch.device):
    """Load an inference-only checkpoint created for this release."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported checkpoint format")
    family = payload.get("model_family")
    if family not in MODEL_FAMILIES:
        raise ValueError(f"{path}: unknown model family {family!r}")
    model = MODEL_FAMILIES[family](**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def checkpoint_paths(
    checkpoint_dir: str | Path,
    task: str,
    manifest_path: str | Path,
) -> list[Path]:
    task = task.upper()
    manifest = json.loads(Path(manifest_path).read_text())
    names = [item["filename"] for item in manifest["checkpoints"] if item["task"] == task]
    if len(names) != 8:
        raise ValueError(f"manifest contains {len(names)} {task} checkpoints, expected 8")
    paths = [Path(checkpoint_dir) / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint files:\n  " + "\n  ".join(missing))
    return paths


@torch.no_grad()
def ensemble_predict(
    data,
    paths: Iterable[str | Path],
    device: str | torch.device,
) -> np.ndarray:
    """Mean the residue probabilities from the released checkpoints."""
    data = data.to(device)
    probabilities = []
    for path in paths:
        model = load_checkpoint(path, device)
        output = model(data)
        logits = output["node_logits"] if isinstance(output, dict) else output.squeeze(-1)
        probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
        del model, output, logits
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return np.stack(probabilities, axis=0).mean(axis=0)
