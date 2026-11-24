"""The optional additive dataset must work for both training methods and modes."""
import json
from pathlib import Path
import subprocess

import pytest

from test_simple_release import ROOT, fake_python


@pytest.mark.parametrize('tuning', ['full'])
@pytest.mark.parametrize('method', ['standard', 'refwave', 'class_isolated'])
def test_select_new_records_without_changing_image_root(tmp_path, tuning, method):
    env = fake_python(tmp_path)
    env.update(METHOD=method, RECORDS_DIR=str(tmp_path/'records with spaces/refwave_v2_cod'))
    result = subprocess.run(['bash', str(ROOT/f'train_{tuning}.sh'), '2', 'stage1'],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    command = json.loads(Path(env['COMMAND_LOG']).read_text())
    assert command[command.index('--train-data')+1] == env['RECORDS_DIR']+'/stage1.jsonl'
    assert command[command.index('--validation-data')+1] == env['RECORDS_DIR']+'/validation.jsonl'
    assert command[command.index('--data-root')+1] == str(Path(env['WORK_ROOT'])/'dataset')
    assert 'RECORDS_DIR=' in result.stdout
    assert not any('verify' in c or 'audit' in c for c in command)
