# Maintainer upload checklist

These commands are instructions only. They have not been run against the remote repository.

## 1. Replace the old repository contents in a disposable clone

```bash
git clone https://github.com/zpliulab/PNBind.git PNBind_publish
rsync -a --delete --exclude .git/ /absolute/path/to/PNBind_Git_release_20260807_223857/ PNBind_publish/
cd PNBind_publish
python scripts/verify_release.py
git status --short
git diff --stat
```

Review the deletion of the old training scripts and obsolete model files before committing.

## 2. Commit and push only after manual review

```bash
git add -A
git commit -m "Release reproducible PNBind inference and evaluation package"
git push origin main
```

## 3. Upload model weights as release assets

The `.pt` files must not be added to the Git commit. Once the repository commit is online, create a release and attach the 16 files from the separately prepared asset directory:

```bash
gh release create v1.0.0 /absolute/path/to/PNBind_model_assets_20260807_223857/checkpoints/*.pt \
  --repo zpliulab/PNBind \
  --title "PNBind v1.0.0" \
  --notes "Inference-only checkpoints for the eight-model DNA and RNA ensembles."
```

Finally, test the public download path in a fresh clone:

```bash
python scripts/download_checkpoints.py --tag v1.0.0 --task DNA
python scripts/infer_precomputed.py --task DNA --checkpoint-dir checkpoints --check
```

Do not create the GitHub release until the repository owner has chosen a license for the public code and benchmark artifacts.
