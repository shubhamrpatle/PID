#!/usr/bin/env python3
"""Download RefWave records and reuse the dataset's existing image archives."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile

from huggingface_hub import snapshot_download

MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
DATA_REPO = "shubhampatle/Pixel-PIVR-Magnified-v2"
DATA_REVISION = "0ece8e8145324265f5c6287fcca6aba75b050c22"


def extract_archive(path, root):
    """Extract regular images only. Preserve completed files across restarts."""
    root = root.resolve()
    with tarfile.open(path) as archive:
        for member in archive:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not member.name.startswith("images/"):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Non-regular archive member: {member.name}")
            if target.exists() and target.stat().st_size == member.size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            with archive.extractfile(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            os.replace(temporary, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work-root", type=Path, default=Path(__file__).resolve().parents[2] / "refwave-assets"
    )
    p.add_argument("--dataset-revision", default=DATA_REVISION)
    p.add_argument("--include-test", action="store_true")
    args = p.parse_args()
    work = args.work_root.resolve()
    snapshot_download(
        "nvidia/LocateAnything-3B",
        revision=MODEL_REVISION,
        local_dir=work / "LocateAnything-3B",
        ignore_patterns=["assets/*"],
        max_workers=8,
    )
    patterns = ["refwave_v1/*", "archives/images-train-*.tar", "archives/images-validation-*.tar"]
    if args.include_test:
        patterns += ["archives/images-test-*.tar", "annotations/test/**"]
    root = work / "dataset"
    snapshot_download(
        DATA_REPO,
        repo_type="dataset",
        revision=args.dataset_revision,
        local_dir=root,
        allow_patterns=patterns,
        max_workers=8,
    )
    for archive in sorted((root / "archives").glob("images-*.tar")):
        if "images-test-" in archive.name and not args.include_test:
            continue
        print("Extracting", archive.name, flush=True)
        extract_archive(archive, root)
    manifest = json.loads((root / "refwave_v1/manifest.json").read_text())
    for stage, entry in manifest["splits"].items():
        print(stage, entry["counts"]["records"], "episodes")
    print("Ready:", root)


if __name__ == "__main__":
    main()
