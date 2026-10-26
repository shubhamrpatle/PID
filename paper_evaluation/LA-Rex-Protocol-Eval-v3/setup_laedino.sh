#!/usr/bin/env bash
# Separate environment: never replace the working LA/Qwen PyTorch installation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIP_CONSTRAINT="$ROOT/requirements-laedino-constraints.txt"
# Keep build dependencies on PyPI, even if the caller uses a CUDA-only index.
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=
CREATE_PYTHON="${CREATE_PYTHON:-python3.10}"
ENV_DIR="${LAEDINO_ENV:-$ROOT/.envs/laedino}"
CODE="${LAEDINO_CODE:-}"
if [[ -z "$CODE" && -f "${RESULTS_DIR:-$ROOT/results}/models.json" ]]; then
  CODE="$("$CREATE_PYTHON" -c 'import json, sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["laedino"]["code"]) / "mmdetection_lae")' \
    "${RESULTS_DIR:-$ROOT/results}/models.json")"
fi
CODE="${CODE:-$ROOT/models/LAE-DINO/mmdetection_lae}"
if [[ ! -f "$CODE/setup.py" ]]; then
  printf '%s\n' 'First run prepare.py models --models laedino (or the complete model list).' >&2
  exit 2
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then "$CREATE_PYTHON" -m venv "$ENV_DIR"; fi
P="$ENV_DIR/bin/python"
"$P" -m pip install --upgrade pip==24.3.1 'setuptools<70' wheel
"$P" -m pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --find-links https://download.pytorch.org/whl/cu121/torch/ \
  --find-links https://download.pytorch.org/whl/cu121/torchvision/
"$P" -m pip install numpy==1.26.4 mmengine==0.10.7 opencv-python==4.11.0.86
# Require the published CUDA extension wheel; do not accidentally build against the other environment.
"$P" -m pip install mmcv==2.1.0 --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
"$P" -m pip install transformers==4.42.3 huggingface-hub==0.23.4 tokenizers==0.19.1 \
  safetensors==0.4.3 scipy==1.13.1 pycocotools==2.0.8 terminaltables six shapely==2.0.7 \
  opencv-python-headless==4.11.0.86 matplotlib==3.8.4 Pillow==11.1.0 ftfy regex einops \
  timm==1.0.27 fairscale==0.4.13 emoji==2.15.0 lvis==0.5.3 yapf==0.43.0 addict==2.4.0 \
  open-clip-torch==2.26.1
# PyPI's package named "clip" is a clipboard tool, not OpenAI CLIP.
"$P" -m pip install --no-deps \
  'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
"$P" -m pip install --no-deps --no-build-isolation -e "$CODE"
printf 'Detector Python: %s\n' "$P"
