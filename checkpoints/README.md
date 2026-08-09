# Checkpoints

The 16 inference-only checkpoint files are distributed as GitHub Release assets, not as ordinary Git objects. Each sanitized file is approximately 72 MiB; keeping the combined 1.2 GiB ensemble out of Git makes cloning and versioning practical.

After release `v1.0.0` has been created, download and verify all files with:

```bash
python scripts/download_checkpoints.py --tag v1.0.0 --task all
```

`manifest.json` records the exact filenames, sizes, SHA-256 digests, tasks, and inference model families. The checkpoint payloads contain only the state dictionary and the model configuration needed for inference; optimizer state, epochs, validation history, paths, and training arguments have been removed.
