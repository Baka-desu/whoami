"""Map statistics kernel (/ugv/map/stats). Clocks and numbers only: no ROS, no files."""

from __future__ import annotations

import json
import math

import pytest

import os

from ugv_localization.mapstats import CLOSURE_LINK_TYPES, MapStats, closure_pairs, path_length, regular_file_size
from ugv_localization.modes import Mode

_KEYS = (
    "keyframes",
    "loop_closures",
    "path_length_m",
    "db_bytes",
    "last_update_age_s",
    "mode",
    "calibration_placeholder",
)
_LINE = [(0.0, 0.0), (3.0, 4.0), (3.0, 10.0)]  # 5 m then 6 m


def _stats(mode: Mode = Mode.MAPPING, placeholder: bool = False) -> MapStats:
    return MapStats(mode, calibration_placeholder=placeholder)


def _snap(s: MapStats, now: float = 0.0, db: int | None = None) -> dict:
    return s.snapshot(now, db)


# --- path_length -----------------------------------------------------------------------------


def test_m1_path_length_of_a_3_4_5_polyline() -> None:
    assert path_length([(0.0, 0.0), (3.0, 4.0)]) == pytest.approx(5.0)
    assert path_length(_LINE) == pytest.approx(11.0)


def test_m2_path_length_of_fewer_than_two_poses_is_zero() -> None:
    assert path_length([]) == 0.0
    assert path_length([(7.0, -2.0)]) == 0.0


def test_m3_path_length_is_the_polyline_not_the_displacement() -> None:
    out_and_back = [(0.0, 0.0), (4.0, 0.0), (0.0, 0.0)]
    assert path_length(out_and_back) == pytest.approx(8.0)


# --- loop closures ----------------------------------------------------------------------------

NEIGHBOR, GLOBAL, LOCAL_SPACE, LOCAL_TIME, USER, VIRTUAL, MERGED, PRIOR, LANDMARK, GRAVITY = range(10)  # rtabmap::Link::Type


def _closures(links: list[tuple[int, int, int]], mode: Mode = Mode.MAPPING) -> int:
    s = _stats(mode)
    s.on_graph([1, 2, 3], _LINE, 100.0, links)
    return _snap(s)["loop_closures"]


def test_m4_a_parked_robot_graph_with_only_neighbour_links_has_no_closures() -> None:
    assert _closures([(1, 2, NEIGHBOR), (2, 3, NEIGHBOR), (3, 3, GRAVITY)]) == 0
    assert _closures([]) == 0


def test_m5_the_three_closure_types_count_and_nothing_else_does() -> None:
    assert _closures([(1, 3, GLOBAL)]) == 1
    assert _closures([(1, 3, GLOBAL), (1, 2, LOCAL_SPACE), (2, 3, USER)]) == 3
    others = [LOCAL_TIME, VIRTUAL, MERGED, PRIOR, LANDMARK, GRAVITY, NEIGHBOR, 10, 97, 99, -1]
    assert _closures([(1, 3, k) for k in others]) == 0


def test_m6_duplicate_and_reversed_pairs_are_one_closure() -> None:
    assert _closures([(1, 3, GLOBAL), (1, 3, GLOBAL), (3, 1, GLOBAL), (3, 1, USER)]) == 1
    assert _closures([(1, 3, GLOBAL), (1, 2, GLOBAL)]) == 2


def test_m7_closures_follow_the_latest_graph_and_a_fresh_database_starts_over() -> None:
    s = _stats()
    s.on_graph([1, 2, 3], _LINE, 100.0, [(1, 3, GLOBAL), (1, 2, LOCAL_SPACE)])
    assert _snap(s)["loop_closures"] == 2
    s.on_graph([1], [(0.0, 0.0)], 110.0, [])  # rtabmap restarted on a new database: ids restart, graph shrinks
    snap = _snap(s, 111.0)
    assert (snap["loop_closures"], snap["keyframes"]) == (0, 1)
    assert snap["last_update_age_s"] == pytest.approx(1.0)


def test_m7b_closure_ids_are_clean_ints_in_a_localize_graph() -> None:
    assert _closures([(1, 3, GLOBAL)], Mode.LOCALIZE) == 1


def test_closure_pairs_helper_and_type_set() -> None:
    assert CLOSURE_LINK_TYPES == frozenset({GLOBAL, LOCAL_SPACE, USER})
    assert closure_pairs([(5, 2, GLOBAL), (2, 5, USER), (4, 4, GRAVITY)]) == 1


# --- last_update_age_s ------------------------------------------------------------------------


def test_m8_age_is_null_before_the_first_graph() -> None:
    assert _snap(_stats(), now=500.0)["last_update_age_s"] is None


def test_m9_age_comes_from_the_injected_clock() -> None:
    s = _stats()
    s.on_graph([1, 2], _LINE[:2], 100.0)
    assert _snap(s, 100.0)["last_update_age_s"] == pytest.approx(0.0)
    assert _snap(s, 103.5)["last_update_age_s"] == pytest.approx(3.5)
    s.on_graph([1, 2, 3], _LINE, 110.0)  # a new keyframe
    assert _snap(s, 112.0)["last_update_age_s"] == pytest.approx(2.0)


def test_m10_an_unchanged_graph_republished_does_not_reset_the_age() -> None:
    s = _stats()
    s.on_graph([1, 2, 3], _LINE, 100.0)
    s.on_graph([1, 2, 3], list(_LINE), 101.0)  # equal values in new containers, as every message is
    s.on_graph((1, 2, 3), tuple(_LINE), 104.0)
    assert _snap(s, 105.0)["last_update_age_s"] == pytest.approx(5.0)


def test_m11_a_moved_pose_or_a_changed_id_list_resets_the_age() -> None:
    s = _stats()
    s.on_graph([1, 2, 3], _LINE, 100.0)
    moved = [(0.0, 0.0), (3.0, 4.0), (3.0, 10.5)]  # loop closure: the optimizer shifts the last pose
    s.on_graph([1, 2, 3], moved, 110.0)
    assert _snap(s, 111.0)["last_update_age_s"] == pytest.approx(1.0)
    s.on_graph([1, 2], moved[:2], 120.0)  # a node was removed
    assert _snap(s, 121.0)["last_update_age_s"] == pytest.approx(1.0)


def test_m12_a_change_in_height_or_heading_alone_is_a_change() -> None:
    # poses carry x, y, z and the quaternion; only x/y feed the path, every component feeds change detection
    s = _stats()
    s.on_graph([1, 2], [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)], 100.0)
    s.on_graph([1, 2], [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0, 0.0, 0.7071, 0.7071)], 105.0)
    assert _snap(s, 106.0)["last_update_age_s"] == pytest.approx(1.0)
    assert _snap(s, 106.0)["path_length_m"] == pytest.approx(1.0)


def test_m13_a_clock_that_ran_backwards_never_gives_a_negative_age() -> None:
    s = _stats()
    s.on_graph([1], [(0.0, 0.0)], 100.0)
    assert _snap(s, 90.0)["last_update_age_s"] == 0.0


# --- graph numbers ----------------------------------------------------------------------------


def test_m14_ids_out_of_order_are_sorted_with_their_poses() -> None:
    s = _stats()
    # message order 3, 1, 2; id order is the (0,0) -> (3,4) -> (3,10) line
    s.on_graph([3, 1, 2], [_LINE[2], _LINE[0], _LINE[1]], 100.0)
    snap = _snap(s, 100.0)
    assert snap["path_length_m"] == pytest.approx(11.0)
    assert snap["keyframes"] == 3
    # the same graph in id order is the same graph: no change
    s.on_graph([1, 2, 3], _LINE, 108.0)
    assert _snap(s, 109.0)["last_update_age_s"] == pytest.approx(9.0)


def test_m15_landmarks_are_not_keyframes() -> None:
    # rtabmap lists tag landmarks in the graph with negative ids; they are not part of the driven path
    s = _stats()
    s.on_graph([-5, 1, 2], [(100.0, 100.0), (0.0, 0.0), (3.0, 4.0)], 100.0)
    snap = _snap(s)
    assert snap["keyframes"] == 2
    assert snap["path_length_m"] == pytest.approx(5.0)


def test_m16_an_empty_graph_is_a_graph() -> None:
    s = _stats()
    s.on_graph([], [], 100.0)
    snap = _snap(s, 102.0)
    assert (snap["keyframes"], snap["path_length_m"], snap["last_update_age_s"]) == (0, 0.0, pytest.approx(2.0))


def test_m17_mismatched_ids_and_poses_are_refused_and_change_nothing() -> None:
    s = _stats()
    s.on_graph([1, 2], _LINE[:2], 100.0)
    with pytest.raises(ValueError):
        s.on_graph([1, 2, 3], _LINE[:2], 105.0)
    with pytest.raises(ValueError):
        s.on_graph([1], [(1.0,)], 105.0)  # a pose needs x and y
    snap = _snap(s, 106.0)
    assert (snap["keyframes"], snap["last_update_age_s"]) == (2, pytest.approx(6.0))


# --- the published dict -----------------------------------------------------------------------


def test_m18_non_finite_floats_become_null() -> None:
    s = _stats()
    s.on_graph([1, 2, 3], [(0.0, 0.0), (math.nan, 1.0), (2.0, 2.0)], 100.0)
    snap = _snap(s, 101.0)
    assert snap["path_length_m"] is None
    assert snap["keyframes"] == 3
    s.on_graph([1, 2], [(0.0, 0.0), (math.inf, 1.0)], 102.0)
    assert _snap(s, 103.0)["path_length_m"] is None
    json.dumps(snap, allow_nan=False)  # would raise on a NaN or inf left in the dict


def test_m19_every_key_is_a_json_scalar_with_the_documented_type() -> None:
    s = _stats(Mode.LOCALIZE, placeholder=True)
    s.on_graph([1, 2, 3], _LINE, 100.0, [(1, 3, GLOBAL)])
    snap = s.snapshot(101.0, 4096)
    assert tuple(snap) == _KEYS
    assert snap == {
        "keyframes": 3,
        "loop_closures": 1,
        "path_length_m": pytest.approx(11.0),
        "db_bytes": 4096,
        "last_update_age_s": None,
        "mode": "localize",
        "calibration_placeholder": True,
    }
    assert type(snap["keyframes"]) is int and type(snap["loop_closures"]) is int and type(snap["db_bytes"]) is int
    assert type(snap["path_length_m"]) is float and snap["last_update_age_s"] is None
    assert all(v is None or type(v) in (int, float, str, bool) for v in snap.values())
    assert json.loads(json.dumps(snap, allow_nan=False)) == snap


def test_m20_before_anything_arrives_the_numbers_are_zero_or_null() -> None:
    snap = _snap(_stats(Mode.MAPPING, placeholder=False), now=10.0, db=None)
    assert snap == {
        "keyframes": 0,
        "loop_closures": 0,
        "path_length_m": 0.0,
        "db_bytes": None,
        "last_update_age_s": None,
        "mode": "mapping",
        "calibration_placeholder": False,
    }


def test_m21_snapshot_does_not_change_the_state() -> None:
    s = _stats()
    s.on_graph([1, 2], _LINE[:2], 100.0)
    first = s.snapshot(101.0, 10)
    assert s.snapshot(101.0, 10) == first
    assert s.snapshot(105.0, 10)["last_update_age_s"] == pytest.approx(5.0)
    assert s.snapshot(101.0, 10) == first


# --- argument checks --------------------------------------------------------------------------


@pytest.mark.parametrize("db", [-1, True, 1.5, "10"])
def test_m22_db_bytes_must_be_a_size_or_none(db: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _snap(_stats(), db=db)  # type: ignore[arg-type]


@pytest.mark.parametrize("now", [math.nan, math.inf, "1"])
def test_m23_the_clock_must_be_a_finite_number(now: object) -> None:
    s = _stats()
    with pytest.raises((TypeError, ValueError)):
        s.on_graph([1], [(0.0, 0.0)], now)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        s.snapshot(now, None)  # type: ignore[arg-type]


def test_m24_mode_and_placeholder_types_are_checked() -> None:
    with pytest.raises(TypeError):
        MapStats("mapping")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MapStats(Mode.MAPPING, calibration_placeholder="no")  # type: ignore[arg-type]


# --- review round 1 ---------------------------------------------------------------------------


def test_localize_mode_has_no_update_age_even_with_a_graph() -> None:
    s = _stats(Mode.LOCALIZE)
    s.on_graph([1, 2], _LINE[:2], 100.0)
    assert _snap(s, 500.0)["last_update_age_s"] is None
    assert _snap(s, 500.0)["keyframes"] == 2


def test_mapping_mode_keeps_its_age() -> None:
    s = _stats(Mode.MAPPING)
    s.on_graph([1, 2], _LINE[:2], 100.0)
    assert _snap(s, 104.0)["last_update_age_s"] == pytest.approx(4.0)


def test_a_graph_with_a_nan_pose_equals_itself() -> None:
    s = _stats()
    nan_graph = [(0.0, 0.0), (math.nan, 1.0)]
    s.on_graph([1, 2], nan_graph, 100.0)
    s.on_graph([1, 2], [(0.0, 0.0), (math.nan, 1.0)], 105.0)
    assert _snap(s, 106.0)["last_update_age_s"] == pytest.approx(6.0)
    s.on_graph([1, 2], [(0.0, 0.0), (2.0, 1.0)], 107.0)  # a real change still resets it
    assert _snap(s, 108.0)["last_update_age_s"] == pytest.approx(1.0)


def test_regular_file_size_of_a_file(tmp_path) -> None:
    f = tmp_path / "rtabmap.db"
    f.write_bytes(b"x" * 123)
    assert regular_file_size(str(f)) == 123


def test_regular_file_size_is_none_for_directory_missing_or_empty_path(tmp_path) -> None:
    assert regular_file_size(str(tmp_path)) is None
    assert regular_file_size(str(tmp_path / "nope.db")) is None
    assert regular_file_size("") is None


def test_regular_file_size_expands_the_home_directory(tmp_path, monkeypatch) -> None:
    (tmp_path / "m.db").write_bytes(b"abcd")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert regular_file_size(os.path.join("~", "m.db")) == 4
