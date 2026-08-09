from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data


def _align_layers(layers: torch.Tensor, n_residues: int) -> torch.Tensor:
    """Prefix-align a (layers, residues, features) tensor to a graph."""
    if layers.shape[1] > n_residues:
        return layers[:, :n_residues, :]
    if layers.shape[1] < n_residues:
        pad = torch.zeros(
            layers.shape[0],
            n_residues - layers.shape[1],
            layers.shape[2],
            dtype=layers.dtype,
        )
        return torch.cat((layers, pad), dim=1)
    return layers


def load_feature_graph(
    graph_path: str | Path,
    esm3_layers_path: Optional[str | Path] = None,
) -> Data:
    """Load a precomputed PNBind graph and optional ESM3 layer features."""
    raw = torch.load(graph_path, map_location="cpu", weights_only=False)
    if isinstance(raw, Data):
        data = raw
    elif isinstance(raw, dict):
        required = (
            "esm_feat",
            "phys_feat",
            "node_vectors",
            "pos",
            "edge_index",
            "edge_attr",
            "edge_vec",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise KeyError(f"{graph_path}: missing graph fields {missing}")
        kwargs = {key: raw[key] for key in required}
        for key in ("y", "protein_id", "coords_5ref", "geo_node_feat"):
            if key in raw:
                kwargs[key] = raw[key]
        data = Data(**kwargs)
    else:
        raise TypeError(f"{graph_path}: unsupported object {type(raw)!r}")

    if esm3_layers_path is not None:
        layer_file = torch.load(
            esm3_layers_path, map_location="cpu", weights_only=False
        )
        if "esm3_layers" not in layer_file:
            raise KeyError(f"{esm3_layers_path}: missing 'esm3_layers'")
        data.esm3_layers = _align_layers(
            layer_file["esm3_layers"], int(data.pos.shape[0])
        ).float()
    return data
