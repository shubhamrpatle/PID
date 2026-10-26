"""Conservative GPU discovery and advisory leases for independent evaluators."""
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess


def query_smi(query):
    result = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=20)
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def probe():
    devices = []
    for row in query_smi('--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu,mig.mode.current'):
        if len(row) != 6:
            raise ValueError('Unexpected nvidia-smi GPU response; not selecting any GPUs')
        index, uuid, free, used, utilization, mig = (v.strip() for v in row)
        if mig.lower() == 'enabled':
            continue  # Physical-GPU scheduling must not bypass MIG allocations.
        try:
            devices.append(dict(index=index, uuid=uuid, free_mb=int(free), used_mb=int(used),
                                utilization=int(utilization), compute_pids=[]))
        except ValueError:
            continue  # Unknown telemetry is not evidence that a GPU is available.
    for row in query_smi('--query-compute-apps=gpu_uuid,pid'):
        if len(row) != 2 or not row[1].strip().isdecimal():
            raise ValueError('Cannot verify GPU process occupancy; not selecting any GPUs')
        for device in devices:
            if device['uuid'] == row[0].strip():
                device['compute_pids'].append(int(row[1]))
    return devices


def allocated(devices):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        if os.environ.get('SLURM_JOB_ID'):
            raise ValueError('Automatic selection in SLURM requires CUDA_VISIBLE_DEVICES from the allocation')
        return devices
    tokens = [v.strip() for v in visible.split(',')]
    if not tokens or any(not (v.isdecimal() or v.startswith('GPU-')) for v in tokens):
        raise ValueError('CUDA_VISIBLE_DEVICES must expose physical GPU indices or UUIDs; MIG is not supported')
    result = {}
    for token in tokens:
        matches = [d for d in devices if d['index'] == token or d['uuid'].startswith(token)]
        if len(matches) > 1:
            raise ValueError('Ambiguous GPU UUID in CUDA_VISIBLE_DEVICES')
        for device in matches:
            result[device['uuid']] = device
    return list(result.values())


class GpuPool:
    def __init__(self, max_gpus=4, min_free_mb=24576, max_used_mb=1024,
                 max_utilization=5, idle_polls=2, lock_dir=None):
        if not 1 <= max_gpus <= 4 or min_free_mb <= 0 or max_used_mb < 0 or idle_polls < 1:
            raise ValueError('Invalid automatic GPU limits (maximum is four GPUs)')
        if not 0 <= max_utilization <= 100:
            raise ValueError('Invalid utilization threshold')
        self.max_gpus = max_gpus
        self.min_free_mb = min_free_mb
        self.max_used_mb = max_used_mb
        self.max_utilization = max_utilization
        self.idle_polls = idle_polls
        self.idle = {}
        self.lock_dir = Path(lock_dir) if lock_dir else Path('/tmp') / f'rs-vlm-eval-gpus-{os.getuid()}'
        self.lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    def free(self, device):
        return (not device['compute_pids'] and device['free_mb'] >= self.min_free_mb
                and device['used_mb'] <= self.max_used_mb
                and device['utilization'] <= self.max_utilization)

    def available(self, used, eligible_uuids=None):
        devices = allocated(probe())
        if eligible_uuids is not None:
            devices = [d for d in devices if d['uuid'] in eligible_uuids]
        self.idle = {d['uuid']: self.idle.get(d['uuid'], 0) + 1
                     if d['uuid'] not in used and self.free(d) else 0 for d in devices}
        candidates = [d for d in devices if self.idle[d['uuid']] >= self.idle_polls]
        candidates.sort(key=lambda d: (-d['free_mb'], int(d['index'])))
        return candidates[:max(0, self.max_gpus - len(used))]

    def claim(self, device):
        lease = (self.lock_dir / (device['uuid'] + '.lock')).open('a+')
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            return None
        try:
            current = next((d for d in allocated(probe()) if d['uuid'] == device['uuid']), None)
            if current is None or not self.free(current):
                self.idle[device['uuid']] = 0
                lease.close()
                return None
            lease.seek(0)
            lease.truncate()
            json.dump(dict(pid=os.getpid(), gpu=current), lease)
            lease.flush()
            return lease
        except BaseException:
            lease.close()
            raise
