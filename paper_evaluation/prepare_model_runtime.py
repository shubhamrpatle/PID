#!/usr/bin/env python3
"""Build the patched LocateAnything-3B runtime used by the paper's evaluations.

The evaluations ran on nvidia/LocateAnything-3B with three source files patched.
This script reproduces that runtime exactly, without redistributing NVIDIA code:

  1. take the released model at the pinned revision, from an existing download
     or from Hugging Face;
  2. copy its small code and configuration files into the evaluation workspace,
     and link its weight shards instead of copying them;
  3. apply the three patches in model_patches/LocateAnything-3B/;
  4. check all 21 runtime files against Final-RefWave-Eval/runtime_hashes.json,
     the integrity manifest the evaluation code itself enforces.

It stops on any mismatch rather than continuing with a different model.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ID = "nvidia/LocateAnything-3B"
# Same revision as tools/download_refwave_assets.py. The 21 runtime files are
# identical at the evaluation reference revision 7a81d810; only README.md,
# batch_infer.py and pyproject.toml differ between the two.
REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
TARGET = HERE / "Final-Controls-Eval/workspace/Zero_shot_ anlysis_LA/model/LocateAnything-3B/checkpoint"
PATCHES = HERE / "model_patches/LocateAnything-3B"
MANIFEST = HERE / "Final-RefWave-Eval/runtime_hashes.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_hashes():
    data = json.loads(MANIFEST.read_text())
    out = {}
    for key, value in data.items():
        if "/model/" in key:
            out[key.split("checkpoint/")[-1]] = value["sha256"] if isinstance(value, dict) else value
    if len(out) != 21:
        raise SystemExit(f"Expected 21 model runtime entries in {MANIFEST}, found {len(out)}")
    return out


def obtain_source(source):
    if source:
        source = Path(source).resolve()
        if not source.is_dir():
            raise SystemExit(f"--source is not a directory: {source}")
        return source
    from huggingface_hub import snapshot_download
    print(f"Downloading {REPO_ID} at {REVISION}", flush=True)
    return Path(snapshot_download(REPO_ID, revision=REVISION))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="Existing LocateAnything-3B download at the pinned revision, "
                                    "for example $WORK_ROOT/LocateAnything-3B. Downloads if omitted.")
    p.add_argument("--force", action="store_true", help="Rebuild even if the target already verifies.")
    args = p.parse_args()

    expected = expected_hashes()
    if TARGET.is_dir() and not args.force:
        if all((TARGET / n).is_file() and sha256(TARGET / n) == h for n, h in expected.items()):
            print(f"Model runtime already verified: {TARGET}")
            return

    source = obtain_source(args.source)
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    for item in sorted(source.iterdir()):
        if item.name.startswith(".") or item.is_dir():
            continue
        if item.suffix == ".safetensors":
            (TARGET / item.name).symlink_to(item.resolve())
        else:
            shutil.copy2(item, TARGET / item.name)

    for patch in sorted(PATCHES.glob("*.patch")):
        result = subprocess.run(["patch", "-p1", "--forward", "--silent", "-i", str(patch)],
                                cwd=TARGET, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"Failed to apply {patch.name}:\n{result.stdout}{result.stderr}\n"
                             f"Check that --source is {REPO_ID} at revision {REVISION}.")
        print(f"applied {patch.name}")

    bad = [n for n, h in expected.items() if not (TARGET / n).is_file() or sha256(TARGET / n) != h]
    if bad:
        raise SystemExit("Runtime does not match the evaluated version for: " + ", ".join(bad))
    weights = sorted(p.name for p in TARGET.glob("*.safetensors"))
    print(f"Verified 21/21 model runtime files against {MANIFEST.name}.")
    print(f"Weight shards linked: {len(weights)}" + ("" if weights else
          "  (none found; inference needs them, so pass --source with the full download)"))
    print(f"Runtime ready: {TARGET}")


if __name__ == "__main__":
    sys.exit(main())
