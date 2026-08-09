#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PNBind inference-only checkpoints from a GitHub Release")
    parser.add_argument("--repo", default="zpliulab/PNBind")
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--task", choices=("DNA", "RNA", "all"), default="all")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text())
    selected = [
        item for item in manifest["checkpoints"]
        if args.task == "all" or item["task"] == args.task
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/{args.repo}/releases/download/{args.tag}"
    for item in selected:
        target = args.output / item["filename"]
        if target.is_file() and sha256(target) == item["sha256"]:
            print(f"OK      {target.name}")
            continue
        url = f"{base}/{item['filename']}"
        print(f"GET     {url}")
        urllib.request.urlretrieve(url, target)
        actual = sha256(target)
        if actual != item["sha256"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {item['filename']}")
        print(f"VERIFIED {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
