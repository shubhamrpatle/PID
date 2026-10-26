#!/usr/bin/env bash
# Standalone uploader: copy just this file to the cluster and run it with bash.
set +x
set -euo pipefail

# Upload destination and source. Do not put your token in this file.
DEFAULT_REPO_ID="CaseStudyRef/RefWave-Cluster-Runs"
DEFAULT_SOURCE="${WORK_ROOT:-$PWD/refwave-assets}/runs"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf '%s\n' \
    'Usage: bash upload_cluster_runs.sh [USER/PUBLIC_BACKUP_REPO [RUNS_DIRECTORY]]' \
    "Default repository: $DEFAULT_REPO_ID" \
    "Default directory: $DEFAULT_SOURCE" \
    'Creates a public Hugging Face dataset repository used as a checkpoint archive.' \
    'If the selected repository is private, it becomes public, including its history.' \
    'Wait for training to finish or stop cleanly before uploading.' \
    'Token: HF_TOKEN, an existing Hugging Face login, or a hidden interactive prompt.' \
    'Optional: UPLOAD_WORKERS=4, UPLOAD_PYTHON=python3, UPLOAD_VENV=/path/to/upload-venv'
  exit 0
fi

if [[ $# -gt 2 ]]; then
  printf 'Usage: bash %s [USER/PUBLIC_BACKUP_REPO [RUNS_DIRECTORY]]\n' "$0" >&2
  exit 2
fi
REPO_ID="${1:-$DEFAULT_REPO_ID}"
SOURCE="${2:-$DEFAULT_SOURCE}"
if [[ ! -d "$SOURCE" ]]; then
  printf 'Folder not found: %s\nRun this script on the cluster that contains this folder.\n' "$SOURCE" >&2
  exit 2
fi

BOOTSTRAP="${UPLOAD_PYTHON:-python3}"
UPLOAD_ENV="${UPLOAD_VENV:-${XDG_CACHE_HOME:-$HOME/.cache}/refwave-hf-upload-0.36.2}"
if [[ ! -x "$UPLOAD_ENV/bin/python" ]]; then
  "$BOOTSTRAP" -m venv "$UPLOAD_ENV"
fi
# This environment is only for uploads. Do not change training dependencies.
"$UPLOAD_ENV/bin/python" -m pip install --disable-pip-version-check --quiet 'huggingface_hub==0.36.2'
export HF_DEBUG=0
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_VERBOSITY=info

"$UPLOAD_ENV/bin/python" - "$REPO_ID" "$SOURCE" "${UPLOAD_WORKERS:-4}" <<'PY'
import getpass
import json
import os
from pathlib import Path
import sys
import warnings

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import validate_repo_id

repo_id, folder, workers = sys.argv[1:]
validate_repo_id(repo_id)
if len(repo_id.split('/')) != 2:
    raise SystemExit('Use an explicit repository name: USER/REPO')
workers = int(workers)
if workers < 1:
    raise SystemExit('UPLOAD_WORKERS must be at least 1.')
source = Path(folder).expanduser().resolve()
if not source.is_dir():
    raise SystemExit(f'Folder not found: {source}')

# Large-folder resume metadata belongs to one destination repository.
marker = source / '.cache/refwave-upload/destination.json'
destination = {'repo_id': repo_id, 'repo_type': 'dataset', 'revision': 'main'}
if marker.exists() and json.loads(marker.read_text()) != destination:
    raise SystemExit('This folder has upload metadata for another repository. '
                     'Use the original destination to resume; see docs/UPLOAD_RUNS.md.')

token = os.environ.get('HF_TOKEN') or get_token()
if not token:
    try:
        with warnings.catch_warnings(), open('/dev/tty', 'w') as tty:
            # Never allow getpass to fall back to visible terminal input.
            warnings.simplefilter('error', getpass.GetPassWarning)
            token = getpass.getpass('Hugging Face token (write access, hidden): ', stream=tty).strip()
    except (OSError, EOFError, getpass.GetPassWarning):
        raise SystemExit('Hidden token entry is unavailable; upload was not started. '
                         'Use an interactive terminal or a job secret environment.')
if not token:
    raise SystemExit('No token provided.')

api = HfApi(token=token)
# Deliberately do not call login(): this token is used only by this process.
identity = api.whoami()
print(f"Authenticated as: {identity['name']}", flush=True)
print('PUBLIC UPLOAD: checkpoints, logs and results will be accessible to everyone.', flush=True)
print('Review files for secrets before upload. Public visibility does not bypass storage quotas.', flush=True)
api.create_repo(repo_id=repo_id, repo_type='dataset', private=False, exist_ok=True)
if api.repo_info(repo_id=repo_id, repo_type='dataset').private:
    print(f'Making {repo_id} public, including all existing files and history.', flush=True)
    api.update_repo_settings(repo_id=repo_id, repo_type='dataset', private=False)
    if api.repo_info(repo_id=repo_id, repo_type='dataset').private:
        raise SystemExit('Repository is still private; upload was not started.')

marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(json.dumps(destination, indent=2) + '\n')
# Keep checkpoints, optimizer/RNG states, metrics, configuration and W&B logs.
# Exclude only metadata caches, incomplete temporary files and credential files.
excluded_names = [
    '.git/*', '.cache/*', '__pycache__/*', '*.pyc',
    '*.tmp', '*.tmp.*', '*.partial', '*.partial/*', '*.partial.*',
    '.env', '.env.*', '.netrc', '.ssh/*', 'id_rsa', 'id_ed25519',
    '*.pem', '*.key', 'token', 'stored_tokens',
]
ignore_patterns = [pattern for name in excluded_names for pattern in (name, '**/' + name)]
print(f'Source: {source}', flush=True)
print(f'Destination: https://huggingface.co/datasets/{repo_id}/tree/main', flush=True)
print('The CONTENTS of runs/ go at the repository root; subdirectories stay unchanged.', flush=True)
print('Upload only while these files are stable. This is not a live checkpoint snapshot.', flush=True)
print('Training is not stopped or modified. No checkpoints are deleted.', flush=True)
print('Review logs for secrets before upload; filename exclusions are not a secret scanner.', flush=True)
print('Interrupted? Rerun the same command with this folder and repository.', flush=True)
api.upload_large_folder(
    repo_id=repo_id, repo_type='dataset', folder_path=source, private=False,
    ignore_patterns=ignore_patterns, num_workers=workers,
    print_report=True, print_report_every=30,
)
print(f'UPLOAD COMPLETE: https://huggingface.co/datasets/{repo_id}/tree/main', flush=True)
print('This backs up files still present locally, not checkpoints previously removed by retention.', flush=True)
PY
