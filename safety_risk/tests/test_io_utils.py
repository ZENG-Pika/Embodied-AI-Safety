import json

import pytest

from safety_risk.io_utils import atomic_write_json


def test_atomic_write_json_publishes_valid_compact_json(tmp_path):
    path = tmp_path / "sim_raw_gt.json"
    atomic_write_json(path, {"unit": "m", "values": [1, 2, 3]})
    assert json.loads(path.read_text()) == {"unit": "m", "values": [1, 2, 3]}
    assert not (tmp_path / "sim_raw_gt.json.tmp").exists()
    assert "\n" not in path.read_text()


def test_atomic_write_json_preserves_old_report_on_failure(tmp_path):
    path = tmp_path / "sim_raw_gt.json"
    path.write_text('{"old":true}')
    circular = []
    circular.append(circular)
    with pytest.raises(ValueError):
        atomic_write_json(path, circular)
    assert json.loads(path.read_text()) == {"old": True}
    assert not (tmp_path / "sim_raw_gt.json.tmp").exists()
