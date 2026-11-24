#!/usr/bin/env bash
#SBATCH --job-name=refwave
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=refwave-%j.log
set -euo pipefail
# Submit from RefWave/. Account/partition and GPU resource syntax are site-specific.
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
srun --ntasks=1 bash "$REPO_ROOT/train.sh" "${GPUS:-8}" "${STAGES:-both}"
