#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    failures = []
    required = (
        "README.md",
        "benchmarks/config.json",
        "checkpoints/manifest.json",
        "results/pnbind_expected.csv",
        "scripts/reproduce_table2.py",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            failures.append(f"missing {relative}")

    checksum_file = ROOT / "release/FILE_SHA256SUMS"
    if not checksum_file.is_file():
        failures.append("missing release/FILE_SHA256SUMS")
    else:
        for line in checksum_file.read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            path = ROOT / relative
            if not path.is_file():
                failures.append(f"checksum target missing: {relative}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                failures.append(f"checksum mismatch: {relative}")

    forbidden_patterns = {
        "absolute workspace path": re.compile("/home/" + "(?:data1/)?ghd"),
        "credential-like assignment": re.compile(
            r"(?i)(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]+"
        ),
    }
    text_suffixes = {".py", ".md", ".json", ".txt", ".csv", ".yml", ".yaml", ".toml", ".cff"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(errors="replace")
        for label, pattern in forbidden_patterns.items():
            if pattern.search(text):
                failures.append(f"{path.relative_to(ROOT)}: {label}")

    oversized = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size > 95 * 1024 * 1024
    ]
    if oversized:
        failures.append(f"files exceed the conservative 95 MiB Git limit: {oversized}")

    reproduction = subprocess.run(
        [sys.executable, str(ROOT / "scripts/reproduce_table2.py"), "--check"],
        cwd=ROOT,
        check=False,
    )
    if reproduction.returncode:
        failures.append("Table 2 reproduction failed")

    if failures:
        print("RELEASE CHECK FAILED", file=sys.stderr)
        print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
        return 1
    print("RELEASE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
