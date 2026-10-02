"""ROS message -> map store source conversions (ugv_api/mapsources.py). Plain Python and numpy, no rclpy.

Every expectation is computed independently of the module under test (hand-worked pinhole numbers, an
explicit rotation matrix) or by the codec's own decoders, so a wrong conversion cannot agree with itself.
"""

from __future__ import annotations

import array
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from ugv_api import mapcodec as codec
from ugv_api import mapsources as ms
from ugv_api.app import create_app
from ugv_api.goals import yaw_to_quaternion

F32 = codec.POINTFIELD_FLOAT32
EPOCH, SEQ, STAMP = 7, 3, 1234.5


# ----------------------------------------------------------------------------------------------- helpers


def pack_rows(columns: list[np.ndarray], *, point_step: int, offsets: list[int]) -> bytes:
    """PointCloud2 payload: each float32 column written at its byte offset in a `point_step` record."""
    n = len(columns[0])
    buf = np.zeros((n, point_step), dtype=np.uint8)
    for col, offset in zip(columns, offsets):
        buf[:, offset : offset + 4] = np.ascontiguousarray(col, dtype="<f4").view(np.uint8).reshape(n, 4)
    return buf.tobytes()


def rgb_word(rgb: np.ndarray) -> np.ndarray:
    """uint8 (N, 3) -> the packed 0x00RRGGBB word as float32 bits, how RTAB-Map stores `rgb`."""
    word = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
    return word.view(np.float32)


# ------------------------------------------------------------------------------------------------- cloud


def test_cloud_source_is_exactly_what_cloud_view_takes_and_decodes_to_the_published_points():
    pts = np.array([[0, 0, 0], [1, 2, 3], [-1, 0.5, 2]], dtype=np.float32)
    col = np.array([[255, 0, 0], [0, 255, 0], [1, 2, 3]], dtype=np.uint8)
    # RTAB-Map layout: x y z, 4 bytes of padding, rgb at 16, 32-byte records
    raw = pack_rows([pts[:, 0], pts[:, 1], pts[:, 2], rgb_word(col)], point_step=32, offsets=[0, 4, 8, 16])
    fields = [("x", 0, F32, 1), ("y", 4, F32, 1), ("z", 8, F32, 1), ("rgb", 16, F32, 1)]
    src = ms.cloud_source(fields=fields, point_step=32, n_points=3, is_bigendian=False, data=array.array("B", raw))

    assert set(src) == {"fields", "point_step", "n_points", "is_bigendian", "data"}
    assert src["fields"] == fields and src["point_step"] == 32 and src["n_points"] == 3
    assert src["is_bigendian"] is False
    xyz, rgb = codec.cloud_view(**src)
    assert np.array_equal(xyz, pts) and np.array_equal(rgb, col)


@pytest.mark.parametrize("make", [bytes, bytearray, lambda b: array.array("B", b)], ids=["bytes", "bytearray", "array"])
def test_cloud_source_holds_a_read_only_view_not_a_copy(make):
    raw = bytes(range(48))
    buffer = make(raw)
    src = ms.cloud_source(fields=[("x", 0, F32, 1)], point_step=12, n_points=4, is_bigendian=False, data=buffer)
    data = src["data"]
    assert isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.shape == (48,)
    assert data.flags.writeable is False  # sources are immutable once stored
    assert np.shares_memory(data, np.frombuffer(buffer, dtype=np.uint8))  # no copy of up to tens of MB
    assert bytes(data) == raw


def test_cloud_source_accepts_a_numpy_buffer_and_normalises_field_tuples():
    data = np.arange(24, dtype=np.uint8)
    src = ms.cloud_source(fields=[["x", 0, 7, 1]], point_step=12, n_points=2, is_bigendian=0, data=data)
    assert src["fields"] == [("x", 0, 7, 1)] and src["is_bigendian"] is False
    assert np.shares_memory(src["data"], data)


def padded_rows_kw(**over):
    """A 3 x 2 organised cloud of 12-byte points. Packed rows are 36 bytes; `row_step` says otherwise."""
    kw = dict(fields=[("x", 0, F32, 1), ("y", 4, F32, 1), ("z", 8, F32, 1)], point_step=12, n_points=6,
              is_bigendian=False, data=bytes(80), width=3, height=2)
    kw.update(over)
    return kw


def test_cloud_source_refuses_an_organised_cloud_whose_rows_are_padded():
    # data is read as n_points * point_step packed bytes; a row_step of 40 means 4 padding bytes after each row, so
    # every point of the second row would be read 4 bytes early
    with pytest.raises(ValueError, match="row_step"):
        ms.cloud_source(**padded_rows_kw(row_step=40))
    assert ms.cloud_source(**padded_rows_kw(row_step=36))["n_points"] == 6  # packed rows are fine


def test_cloud_source_does_not_look_at_row_step_of_a_one_row_cloud():
    # height 1: there is no second row to misplace, and row_step of such a cloud is often 0 or the full buffer
    one_row = padded_rows_kw(n_points=3, width=3, height=1, data=bytes(36))
    assert ms.cloud_source(**one_row, row_step=0)["n_points"] == 3
    assert ms.cloud_source(**one_row, row_step=999)["n_points"] == 3
    assert ms.cloud_source(**one_row)["n_points"] == 3  # and callers that do not know it still work


def test_cloud_source_refuses_a_payload_shorter_than_its_points():
    with pytest.raises(ValueError, match="fewer than"):
        ms.cloud_source(fields=[("x", 0, F32, 1)], point_step=12, n_points=4, is_bigendian=False, data=bytes(47))
    assert ms.cloud_source(fields=[("x", 0, F32, 1)], point_step=12, n_points=4, is_bigendian=False,
                           data=bytes(48))["n_points"] == 4


# ------------------------------------------------------------------------------------------- trajectory


def test_trajectory_source_is_an_n_by_7_float32_array_in_pose_order():
    rows = [(0, 0, 0, 0, 0, 0, 1), (3, 4, 0, 0, 0, 1, 0), (3, 4, 12, 0.5, 0.5, 0.5, 0.5)]
    out = ms.trajectory_source(rows)
    assert out.dtype == np.float32 and out.shape == (3, 7) and out.flags.c_contiguous
    assert out.tolist() == [[float(v) for v in r] for r in rows]
    decoded = codec.decode_trajectory(codec.encode_trajectory(out, epoch=EPOCH, seq=SEQ, stamp_s=STAMP))
    assert decoded["count"] == 3 and decoded["length_m"] == pytest.approx(17.0)


def test_trajectory_source_of_an_empty_path_is_zero_rows_not_an_error():
    out = ms.trajectory_source([])
    assert out.shape == (0, 7) and out.dtype == np.float32


def test_trajectory_source_rejects_rows_that_are_not_pose_sized():
    with pytest.raises(ValueError):
        ms.trajectory_source([(0, 0, 0)])


# ------------------------------------------------------------------------------------------ quaternions


@pytest.mark.parametrize("yaw", [0.0, 0.5, -0.5, math.pi / 2, -math.pi / 2, 3.0, -3.0])
def test_quaternion_yaw_recovers_the_yaw_the_goal_code_encodes(yaw):
    assert ms.quaternion_yaw(*yaw_to_quaternion(yaw)) == pytest.approx(yaw, abs=1e-12)


def test_quaternion_yaw_ignores_roll_and_pitch():
    yaw, pitch, roll = 0.7, 0.2, -0.3
    cy, sy, cp, sp, cr, sr = (math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(roll / 2), math.sin(roll / 2))
    q = (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
         cr * cp * cy + sr * sp * sy)  # ZYX Euler, the ROS convention
    assert ms.quaternion_yaw(*q) == pytest.approx(yaw, abs=1e-9)


def test_quaternion_yaw_of_the_identity_and_a_zero_quaternion():
    assert ms.quaternion_yaw(0, 0, 0, 1) == 0.0
    assert ms.quaternion_yaw(0, 0, 0, 0) == 0.0  # an unset orientation field means "no rotation"


# ----------------------------------------------------------------------------------------------- grid


def test_grid_source_is_the_shape_encode_grid_takes_with_the_yaw_from_the_origin_quaternion():
    values = [-1, 0, 10, 20, 30, 40]
    data = array.array("b", values)
    q = yaw_to_quaternion(0.5)
    src = ms.grid_source(data=data, width=3, height=2, resolution=0.25, origin_x=-0.5, origin_y=1.0, origin_q=q)
    assert set(src) == {"cells", "resolution", "origin_xy", "origin_yaw"}
    assert src["cells"].dtype == np.int8 and src["cells"].shape == (2, 3)
    assert src["cells"].tolist() == [[-1, 0, 10], [20, 30, 40]]  # row-major, row = y
    assert src["cells"].flags.writeable is False
    assert src["resolution"] == 0.25 and src["origin_xy"] == (-0.5, 1.0)
    assert src["origin_yaw"] == pytest.approx(0.5)
    out = codec.decode_grid(codec.encode_grid(epoch=EPOCH, seq=SEQ, stamp_s=STAMP, **src))
    assert out["origin_yaw"] == pytest.approx(0.5, abs=1e-6) and out["cells"].tolist() == src["cells"].tolist()


def test_grid_source_takes_a_plain_list_too():
    src = ms.grid_source(data=[0, 100], width=2, height=1, resolution=1.0, origin_x=0, origin_y=0,
                         origin_q=(0, 0, 0, 1))
    assert src["cells"].tolist() == [[0, 100]] and src["origin_yaw"] == 0.0


def test_grid_source_refuses_a_size_that_disagrees_with_the_data():
    with pytest.raises(ValueError):
        ms.grid_source(data=array.array("b", [0] * 5), width=3, height=2, resolution=1.0, origin_x=0, origin_y=0,
                       origin_q=(0, 0, 0, 1))


# -------------------------------------------------------------------------------------------- depth


def test_depth_source_is_a_float32_h_by_w_copy_with_nan_holes_kept():
    h, w = 3, 4
    image = np.arange(h * w, dtype=np.float32).reshape(h, w) / 4
    image[1, 2] = np.nan
    buffer = array.array("B", image.tobytes())
    src = ms.depth_source(encoding="32FC1", height=h, width=w, step=w * 4, is_bigendian=False, data=buffer)
    assert set(src) == {"depth_m"}
    d = src["depth_m"]
    assert d.dtype == np.float32 and d.shape == (3, 4) and d.flags.c_contiguous and d.flags.writeable is False
    assert np.array_equal(d, image, equal_nan=True)
    assert not np.shares_memory(d, np.frombuffer(buffer, dtype=np.uint8))


def test_depth_source_honours_row_padding():
    h, w, step = 2, 3, 20  # 8 padding bytes per row
    rows = np.zeros((h, step), dtype=np.uint8)
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    rows[:, : w * 4] = values.view(np.uint8).reshape(h, w * 4)
    rows[:, w * 4 :] = 0xAB
    src = ms.depth_source(encoding="32FC1", height=h, width=w, step=step, is_bigendian=False, data=rows.tobytes())
    assert src["depth_m"].tolist() == values.tolist()


def test_depth_source_reads_big_endian_images():
    values = np.array([[1.5, 2.5]], dtype=np.float32)
    src = ms.depth_source(encoding="32FC1", height=1, width=2, step=8, is_bigendian=True,
                          data=values.astype(">f4").tobytes())
    assert src["depth_m"].tolist() == [[1.5, 2.5]]


@pytest.mark.parametrize(
    "kw",
    [
        dict(encoding="16UC1"),  # millimetres: not this layer's unit
        dict(encoding="mono8"),
        dict(step=4),  # a row is 3 floats, 4 bytes is not enough
        dict(data=b"\x00" * 23),  # one byte short
        dict(width=0),
        dict(height=0),
    ],
)
def test_depth_source_refuses_anything_but_a_complete_32fc1_image(kw):
    args = dict(encoding="32FC1", height=2, width=3, step=12, is_bigendian=False, data=b"\x00" * 24)
    args.update(kw)
    with pytest.raises(ValueError):
        ms.depth_source(**args)


# ------------------------------------------------------------------------------------ intrinsics


K = (10.0, 0.0, 4.0, 0.0, 10.0, 2.0, 0.0, 0.0, 1.0)


def test_intrinsics_reads_fx_fy_cx_cy_from_the_row_major_k():
    assert ms.intrinsics(K, width=8, height=6) == (10.0, 10.0, 4.0, 2.0)


def test_intrinsics_scales_k_when_the_image_is_a_resized_calibration_frame():
    assert ms.intrinsics(K, width=4, height=3, info_width=8, info_height=6) == (5.0, 5.0, 2.0, 1.0)
    assert ms.intrinsics(K, width=8, height=6, info_width=8, info_height=6) == (10.0, 10.0, 4.0, 2.0)
    assert ms.intrinsics(K, width=8, height=6, info_width=0, info_height=0) == (10.0, 10.0, 4.0, 2.0)  # unknown


@pytest.mark.parametrize(
    "k",
    [(0.0,) * 9, (math.nan, 0, 4, 0, 10, 2, 0, 0, 1), (10, 0, 4, 0, -10, 2, 0, 0, 1), (10, 0, 4), (10, 0, math.inf, 0, 10, 2, 0, 0, 1)],
    ids=["all-zero", "nan", "negative-fy", "short", "inf-cx"],
)
def test_intrinsics_rejects_a_missing_or_fake_k(k):
    assert ms.intrinsics(k, width=8, height=6) is None


# -------------------------------------------------------------------------------------- back-projection


INTR = (10.0, 10.0, 4.0, 2.0)


def test_backproject_lands_each_pixel_where_the_pinhole_model_puts_it():
    depth = np.full((6, 8), 2.0, dtype=np.float32)
    pts = ms.backproject(depth, INTR, stride=2, min_range_m=0.3, max_range_m=8.0)
    # stride 2 keeps columns 0 2 4 6 and rows 0 2 4: 12 pixels, row-major
    assert pts.dtype == np.float32 and pts.shape == (12, 3)
    expect = [((u - 4.0) * 2.0 / 10.0, (v - 2.0) * 2.0 / 10.0, 2.0) for v in (0, 2, 4) for u in (0, 2, 4, 6)]
    assert np.allclose(pts, np.array(expect, dtype=np.float32), atol=1e-6)
    assert pts[6].tolist() == pytest.approx([0.0, 0.0, 2.0])  # pixel (4, 2) is on the optical axis: row 1, col 2


def test_backproject_stride_picks_every_nth_pixel():
    depth = np.full((6, 8), 1.0, dtype=np.float32)
    assert len(ms.backproject(depth, INTR, stride=1, min_range_m=0.3, max_range_m=8.0)) == 48
    assert len(ms.backproject(depth, INTR, stride=4, min_range_m=0.3, max_range_m=8.0)) == 4  # cols 0 4, rows 0 4
    assert len(ms.backproject(depth, INTR, stride=3, min_range_m=0.3, max_range_m=8.0)) == 6  # cols 0 3 6, rows 0 3


def test_backproject_gates_on_range_inclusively_and_drops_holes():
    depth = np.array([[0.1, 0.3, 1.0, 8.0, 8.5, np.nan, np.inf, -1.0, 0.0]], dtype=np.float32)
    pts = ms.backproject(depth, (10.0, 10.0, 0.0, 0.0), stride=1, min_range_m=0.3, max_range_m=8.0)
    assert pts[:, 2].tolist() == pytest.approx([0.3, 1.0, 8.0])  # 0.1 too near, 8.5 too far, the rest holes
    assert pts[:, 0].tolist() == pytest.approx([1 * 0.3 / 10, 2 * 1.0 / 10, 3 * 8.0 / 10])  # keeps its own column


def test_backproject_of_an_all_hole_image_is_an_empty_n_by_3():
    pts = ms.backproject(np.full((4, 4), np.nan, dtype=np.float32), INTR, stride=1, min_range_m=0.3, max_range_m=8.0)
    assert pts.shape == (0, 3) and pts.dtype == np.float32


def test_backproject_does_not_modify_its_input():
    depth = np.full((6, 8), 2.0, dtype=np.float32)
    depth.flags.writeable = False
    ms.backproject(depth, INTR, stride=2, min_range_m=0.3, max_range_m=8.0)


# ---------------------------------------------------------------------------------- transform_points

# optical (x right, y down, z forward) -> body (x forward, y left, z up): body = (z, -x, -y)
Q_OPTICAL_TO_BODY = (-0.5, 0.5, -0.5, 0.5)


def test_transform_points_rotates_then_translates_by_a_yaw_and_an_offset():
    pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    out = ms.transform_points(pts, (1.0, 2.0, 3.0), yaw_to_quaternion(math.pi / 2))
    # +90 degrees about z: x -> y, y -> -x, z -> z
    assert out.dtype == np.float32
    assert np.allclose(out, [[1, 3, 3], [0, 2, 3], [1, 2, 4]], atol=1e-6)


def test_a_camera_pixel_lands_at_the_expected_map_point():
    """Camera 0.1 m ahead and 0.3 m above the base, base at (1, 2, 0.25) turned 90 degrees left: a point 2 m in
    front of the camera and 0.4 m below its axis ends up where the chain of frames says."""
    depth = np.full((6, 8), 2.0, dtype=np.float32)
    cam = ms.backproject(depth, INTR, stride=4, min_range_m=0.3, max_range_m=8.0)  # pixels (0,0) (4,0) (0,4) (4,4)
    in_base = ms.transform_points(cam, (0.1, 0.0, 0.3), Q_OPTICAL_TO_BODY)
    in_map = ms.transform_points(in_base, (1.0, 2.0, 0.25), yaw_to_quaternion(math.pi / 2))
    # pixel (4, 4): optical (0, 0.4, 2) -> base (2, 0, -0.4) + (0.1, 0, 0.3) = (2.1, 0, -0.1) -> map (1, 4.1, 0.15)
    assert np.allclose(in_map[3], [1.0, 4.1, 0.15], atol=1e-5)
    # pixel (0, 0): optical (-0.8, -0.4, 2) -> base (2, 0.8, 0.4) + (0.1, 0, 0.3) = (2.1, 0.8, 0.7)
    #   -> yaw 90: (-0.8, 2.1, 0.7) + (1, 2, 0.25) = (0.2, 4.1, 0.95)
    assert np.allclose(in_map[0], [0.2, 4.1, 0.95], atol=1e-5)


def test_transform_points_normalises_a_slightly_off_quaternion():
    q = tuple(2.0 * c for c in yaw_to_quaternion(math.pi / 2))  # twice the length
    out = ms.transform_points(np.array([[1, 0, 0]], dtype=np.float32), (0, 0, 0), q)
    assert np.allclose(out, [[0, 1, 0]], atol=1e-6)


def test_transform_points_of_nothing_is_nothing_and_a_zero_quaternion_is_refused():
    assert ms.transform_points(np.zeros((0, 3), np.float32), (1, 2, 3), (0, 0, 0, 1)).shape == (0, 3)
    with pytest.raises(ValueError):
        ms.transform_points(np.zeros((1, 3), np.float32), (0, 0, 0), (0, 0, 0, 0))


# --------------------------------------------------------------------------------------------- stats


def test_parse_stats_returns_the_json_object():
    assert ms.parse_stats('{"keyframes": 12, "mode": "mapping", "calibration_placeholder": false}') == {
        "keyframes": 12, "mode": "mapping", "calibration_placeholder": False}
    assert ms.parse_stats("{}") == {}


@pytest.mark.parametrize("text", ["", "{not json", "[1, 2]", "12", '"text"', "null", "true", "{'a': 1}", "[" * 100_000])
def test_parse_stats_drops_malformed_or_non_object_input(text):
    assert ms.parse_stats(text) is None


# --------------------------------------------------------------------------------------------- stamps


def test_stamp_seconds_uses_the_message_stamp_and_falls_back_for_an_unstamped_one():
    assert ms.stamp_seconds(12, 500_000_000, fallback_s=99.0) == 12.5
    assert ms.stamp_seconds(0, 0, fallback_s=99.0) == 99.0


# --------------------------------------------------------------------------------------------- config


def test_map_config_defaults_are_the_values_the_gateway_ships_with():
    c = ms.MapConfig()
    assert (c.cloud_point_budget, c.cloud_spacing_m) == (500_000, 0.05)
    assert (c.live_stride, c.live_range_min_m, c.live_range_max_m) == (4, 0.3, 8.0)
    assert c.idle_timeout_s == 10.0
    assert (c.cloud_topic, c.trajectory_topic, c.grid_topic) == ("/rtabmap/cloud_map", "/rtabmap/mapPath",
                                                                  "/global_costmap/costmap")
    assert c.depth_topic == "/perception/depth/image"
    assert (c.map_stats_topic, c.perception_stats_topic) == ("/ugv/map/stats", "/ugv/perception/stats")


def test_map_config_app_kwargs_are_exactly_the_tunables_create_app_takes():
    accepted = set(inspect.signature(create_app).parameters)
    kwargs = ms.MapConfig(cloud_point_budget=10, cloud_spacing_m=0.25).app_kwargs()
    assert set(kwargs) == {"cloud_point_budget", "cloud_spacing_m"}
    assert set(kwargs) <= accepted
    assert kwargs["cloud_point_budget"] == 10 and kwargs["cloud_spacing_m"] == 0.25


@pytest.mark.parametrize(
    "bad",
    [
        dict(cloud_point_budget=-1), dict(cloud_spacing_m=0.0), dict(cloud_spacing_m=math.nan),
        dict(live_stride=0), dict(live_range_min_m=-0.1), dict(live_range_max_m=0.0),
        dict(live_range_min_m=5.0, live_range_max_m=5.0), dict(idle_timeout_s=0.0), dict(idle_timeout_s=math.nan),
        dict(stats_stale_s=0.0), dict(cloud_topic=""), dict(depth_topic="relative/topic"),
    ],
)
def test_map_config_refuses_a_bad_value_naming_it(bad):
    with pytest.raises(ValueError, match=next(iter(bad))):
        ms.MapConfig(**bad)


def test_the_shipped_config_file_declares_every_map_parameter_with_the_default_value():
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[1] / "config" / "api.yaml"
    block = yaml.safe_load(path.read_text(encoding="utf-8"))["ugv_api"]["ros__parameters"]["map"]
    defaults = ms.MapConfig()
    from dataclasses import fields

    assert set(block) == {f.name for f in fields(defaults)}
    for f in fields(defaults):
        assert block[f.name] == getattr(defaults, f.name), f.name
        assert type(block[f.name]) is type(getattr(defaults, f.name)), f.name  # ROS parameter types are strict
