# Back Up Cluster Runs to Hugging Face

Copy `upload_cluster_runs.sh` to the cluster. It is standalone: no RefWave
installation, CUDA, GPU allocation, dataset verifier, or training environment
is needed. Python 3.10+ with `venv`, internet access, and a Hugging Face token
with write permission to the destination are required.

**Wait for training/evaluation to finish or stop cleanly first.** The trainer
can replace `best.pt`/`last.pt` and delete older checkpoints during retention.
An upload of a changing directory is not a consistent resume checkpoint.
This script never stops jobs or creates a filesystem snapshot.

## Upload

Run on the cluster where the files exist:

```bash
bash upload_cluster_runs.sh
```

The script already contains these defaults near the top:

```bash
DEFAULT_REPO_ID="CaseStudyRef/RefWave-Cluster-Runs"
DEFAULT_SOURCE="$WORK_ROOT/runs"
```

No command arguments are required. The token is entered at the hidden prompt,
not written into the script. Optional first and second arguments still override
the repository and source folder. The script creates
a **public dataset repository**, used as a backup archive, not a training
dataset or an exported `from_pretrained()` model. If the selected repository is
already private, the script makes it public, including existing files and commit
history. Only choose a repository whose entire contents you intend to publish.
It does not change the published training dataset unless you explicitly select
that dataset as the upload destination; use a separate repository as above.

It uses `HF_TOKEN` or your saved Hugging Face login; otherwise it prompts for a
hidden token. Do not put a token in the script, command arguments, screenshots,
Git, or shared chat. For a batch job, use the cluster's secret environment.
The token supplied at the prompt is not saved by this script. It is not echoed,
passed as a command argument, written to shell history, or saved as a Hugging
Face login. It stays in the upload process's memory for authentication. If the
terminal cannot hide the input, the script stops instead of accepting visible
input. Prefer this prompt over typing `HF_TOKEN=...` in a shell command, which
could put the token in your shell history. An existing saved login is only read;
the script does not create or update it.

The uploader creates a dedicated virtual environment under
`~/.cache/refwave-hf-upload-0.36.2`, with the tested
`huggingface_hub==0.36.2` large-folder uploader. It does **not** upgrade the
training environment or add training prerequisites. Set `UPLOAD_PYTHON` to
your Python executable or `UPLOAD_VENV` to a separate writable environment path
if needed. Do not point `UPLOAD_VENV` at your training environment.

All retained checkpoint files, optimizer states, configurations, training curves,
evaluation outputs, console logs, and W&B files are included. Credentials with
common filenames, temporary/partial files and caches are excluded. Logs can
still contain sensitive text: review before uploading. **Everyone can download
the uploaded files.** The hidden upload token itself is not saved or uploaded.
Already-deleted historical checkpoints cannot be recovered by this upload.

The contents of `runs/` become the repository root:

```text
Hugging Face: CaseStudyRef/RefWave-Cluster-Runs
  refwave_lora_8gpu_<timestamp>_<id>/
    console.log
    stage1/checkpoint-....pt
    stage1/best.pt
    stage1/last.pt
    stage2/...
  refwave_full_8gpu_<timestamp>_<id>/...
  standard_lora_4gpu_<timestamp>_<id>/...
```

File symlinks such as `best.pt` and `last.pt` are uploaded as their target file
contents, not as Unix symlinks. Directory symlinks are not traversed by the pinned
uploader; canonical run/checkpoint directories must be inside the source folder.
This uploads `runs/` only, not sibling datasets, source code or base-model weights.

## Resume and Restore

Rerun the **same command** after interruption. Keep the local `.cache/huggingface/`
upload metadata and use the same destination. Do not run multiple uploaders on
the same folder. To intentionally use a different repository, first finish or
stop the old upload, then move the upload-only `.cache/huggingface/` and
`.cache/refwave-upload/` directories outside `runs/` before starting again.
Do not move checkpoints. Remote-only files are not deleted by this script;
reusing a destination with changed files updates those paths.

To restore the public repository using the upload environment (no login needed):

```bash
~/.cache/refwave-hf-upload-0.36.2/bin/hf download \
  CaseStudyRef/RefWave-Cluster-Runs --repo-type dataset \
  --local-dir /path/to/restored/refwave-assets/runs
```

Choose a new empty destination for restoration. Restoring files alone does not
change absolute paths recorded in training contracts or allow a different GPU
count on resume. Keep the original code, base model and data available separately.

Uploads use four CPU/network workers by default; set `UPLOAD_WORKERS=8` before
the command to change it. They do not use GPUs. Check your account's storage
quota and available local upload-cache space for large checkpoints.

Public visibility does not guarantee unlimited free storage or approval for a
large backup. A 100 GB folder and a single 100 GB file are different cases.
The current [Hub storage policy](https://huggingface.co/docs/hub/storage-limits)
lists a 500 GB single-file hard limit and recommends files below 200 GB; account
quotas still apply. The pinned 0.36.2 uploader can print older 20/50 GB size
warnings. Those are advisory client warnings, not a change to the server's
current policy. No actual 100 GB upload has been tested by this project.

Reference: [Hugging Face upload API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.upload_large_folder).
The pinned version uses local resumable large-folder metadata. Newer Hub versions
may recommend a different upload interface; they are not installed into your
training environment by this script.
