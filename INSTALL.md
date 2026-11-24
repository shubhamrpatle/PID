# Installation

## 1. Get the Code and Create an Environment

Use Linux x86-64 with glibc 2.28 or newer, Python 3.10 or 3.11, and a
CUDA-12.1-compatible NVIDIA driver.
Clone this standalone repository, not the old Pixel-PIVR repository:

```bash
git clone https://github.com/shubhamrpatle/PID.git
cd PID
```

Run the following commands from the cloned `PID/` directory. The code and
Python environment stay here; downloaded images, model weights and training
outputs go in the separate `WORK_ROOT` directory selected in step 2 ([Download the Model and Data](docs/TRAINING.md#2-download-the-model-and-data)).

Choose **one** environment method. No test or data-audit step is required.

### Venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Alternatively, `WORK_ROOT=/scratch/$USER/refwave-assets bash setup.sh` performs
these installations and remembers the storage path in the ignored `.work-root`.
Set `PYTHON_BOOTSTRAP=python3.11` if that is the cluster's available interpreter.
After using `setup.sh`, activate `.venv` and set `PYTHON_BIN` as above too.
Do not reuse the old Pixel-PIVR virtual environment.

### Conda

```bash
conda create -n refwave python=3.10 -y
conda activate refwave
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
export PYTHON_BIN="$(command -v python)"
```

Activate your environment in each new shell or submission environment. Launchers
use an explicit `PYTHON_BIN` first, then an active venv or Conda environment, then
`RefWave/.venv/bin/python` if present, and finally Python on `PATH`. The explicit
export above also works with older checkouts. Conda users do not need a `.venv/`.
Dependencies are pinned to the LA model's APIs. Do not replace PyTorch or
Transformers with arbitrary newer versions in this environment.

The upstream Decord 0.6.0 wheel has an old Python tag in its metadata, so a
manually run `pip check` can report `decord 0.6.0 is not supported on this platform`
([upstream issue](https://github.com/dmlc/decord/issues/366)). Decord imports on the
tested Python 3.10 environment; these image-only launchers do not use video
decoding and do not run that check as a prerequisite.

Next: [download the model and data, and train](docs/TRAINING.md), then [evaluate](docs/EVALUATION.md).
