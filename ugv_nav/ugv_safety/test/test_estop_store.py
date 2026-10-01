"""The persisted e-stop record (pure Python). Fails safe: anything unreadable counts as asserted."""

from __future__ import annotations

import pytest

from ugv_safety.arbiter_core import Level, SafetyArbiter
from ugv_safety.estop_store import EstopLatchStore

from test_arbiter_core import cfg, healthy

S = 1_000_000_000


def test_first_start_with_no_record_is_released(tmp_path):
    assert EstopLatchStore(tmp_path / "estop").load() == (False, "")


def test_asserted_state_survives_a_restart(tmp_path):
    path = tmp_path / "estop"
    assert EstopLatchStore(path).save(True)
    asserted, note = EstopLatchStore(path).load()  # a fresh store object = a restarted process
    assert asserted and "staying asserted" in note


def test_explicit_release_is_remembered(tmp_path):
    path = tmp_path / "estop"
    store = EstopLatchStore(path)
    store.save(True)
    store.save(False)
    assert EstopLatchStore(path).load() == (False, "")


@pytest.mark.parametrize("content", ["", "yes", "2", "true", "garbage\n", "10"])
def test_a_corrupt_record_counts_as_asserted(tmp_path, content):
    path = tmp_path / "estop"
    path.write_text(content, encoding="ascii")
    asserted, note = EstopLatchStore(path).load()
    assert asserted and "corrupt" in note


def test_an_unreadable_record_counts_as_asserted(tmp_path):
    path = tmp_path / "estop"
    path.mkdir()  # a directory where the file should be: reading it fails
    asserted, note = EstopLatchStore(path).load()
    assert asserted and "cannot read" in note


def test_save_creates_missing_directories_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "a" / "b" / "estop"
    assert EstopLatchStore(path).save(True)
    assert path.read_text(encoding="ascii") == "1\n"
    assert [p.name for p in path.parent.iterdir()] == ["estop"]


def test_save_reports_failure_instead_of_raising(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert EstopLatchStore(blocker / "estop").save(True) is False  # parent is a file, cannot create it


def test_a_restarted_arbiter_starting_from_the_record_does_not_drive():
    a = SafetyArbiter(cfg())
    a.on_estop(True, 0)  # what the node does when the record says asserted
    healthy(a, 0)
    a.on_candidate(0.3, 0.0, 0)
    d = a.step(0)
    assert d.level is Level.ESTOP and (d.linear, d.angular) == (0.0, 0.0)
    assert a.estop_asserted is True
