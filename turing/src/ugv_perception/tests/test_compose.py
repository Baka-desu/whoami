"""T07 compose_tick tests — fixtures, no camera, no ROS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import ADAPTER_ID, UNLABELED_NAME, AdapterError, RawSemOutput
from ugv_perception.compose import compose_tick, load_compose_configs
from ugv_perception.confidence.load import load_gates
from ugv_perception.freshness.load import load_freshness
from ugv_perception.remap.load import load_remap

_ROOT = Path(__file__).resolve().parents[3]
_REMAP = _ROOT / "config" / "ontologies" / "yoloe.yaml"
_GATES = _ROOT / "config" / "perception" / "yoloe.yaml"
_FRESH = _ROOT / "config" / "perception" / "port.yaml"

_HW = (4, 4)
_NS = 1_000_000_000
_STAMP = 2_000_000_000
_NOW = _STAMP + _NS // 10  # 0.1s later, within 0.5s max age


def _frame(stamp: int = _STAMP) -> ImageFrame:
    return ImageFrame(
        rgb=np.zeros((_HW[0], _HW[1], 3), dtype=np.uint8),
        stamp_ns=stamp,
        frame_id="camera_optical",
    )


def _raw(*, stamp: int = _STAMP, frame_id: str = "camera_optical", score: float = 0.9) -> RawSemOutput:
    hw = _HW
    return RawSemOutput(
        adapter_id=ADAPTER_ID,
        label_ids=np.ones(hw, dtype=np.int32),  # dirt_path
        raw_scores=np.full(hw, score, dtype=np.float32),
        id_to_name={0: UNLABELED_NAME, 1: "dirt_path"},
        stamp_ns=stamp,
        frame_id=frame_id,
        hw=hw,
    )


class SpyAdapter:
    def __init__(self, raw: RawSemOutput | None = None, error: BaseException | None = None) -> None:
        self.calls = 0
        self._raw = raw if raw is not None else _raw()
        self._error = error

    def infer(self, frame: ImageFrame) -> RawSemOutput:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._raw


@pytest.fixture
def kernels():
    return (
        load_remap(_REMAP),
        load_gates(_GATES),
        load_freshness(_FRESH),
    )


def _tick(kernels, *, frame, adapter, now_ns=_NOW, now_ns_after=None):
    table, gates, fresh = kernels
    return compose_tick(
        frame=frame,
        now_ns=now_ns,
        adapter=adapter,
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
        now_ns_after=now_ns_after,
    )


def test_n1_missing_remap_refuses(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_compose_configs(
            remap_path=tmp_path / "missing.yaml",
            gates_path=_GATES,
            freshness_path=_FRESH,
        )


def test_n2_n14_n15_compose_imports_clean() -> None:
    root = Path(__file__).resolve().parents[1] / "compose"
    banned = ("openvino", "ultralytics", "torch", "rclpy", "cmd_vel")
    for py in root.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert 'if adapter_id == "yoloe"' not in text
        assert "/cmd_vel" not in text
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                for name in banned:
                    assert name not in stripped, f"{py.name}: {stripped}"


def test_n3_mask_stamp_is_sensor_not_now(kernels) -> None:
    spy = SpyAdapter()
    out = _tick(kernels, frame=_frame(), adapter=spy)
    assert out.mask is not None
    assert out.mask.header.stamp_ns == _STAMP
    assert out.mask.header.frame_id == "camera_optical"
    assert out.mask.header.stamp_ns != _NOW


def test_n4_stamp_mismatch_no_mask(kernels) -> None:
    spy = SpyAdapter(raw=_raw(stamp=_STAMP + 1))
    out = _tick(kernels, frame=_frame(), adapter=spy)
    assert spy.calls == 1
    assert out.mask is None
    assert out.decision.degraded is True
    assert out.decision.publish_mask is False


def test_n5_adapter_error_no_mask(kernels) -> None:
    spy = SpyAdapter(error=AdapterError("boom"))
    out = _tick(kernels, frame=_frame(), adapter=spy)
    assert spy.calls == 1
    assert out.mask is None
    assert out.decision.degraded is True


def test_infer_none_is_adapter_error_no_crash(kernels) -> None:
    class NoneAdapter:
        def infer(self, frame):
            return None

    out = _tick(kernels, frame=_frame(), adapter=NoneAdapter())
    assert out.mask is None
    assert out.decision.degraded is True
    assert out.decision.publish_mask is False


def test_infer_missing_stamp_is_adapter_error(kernels) -> None:
    class BadRaw:
        adapter_id = ADAPTER_ID
        frame_id = "camera_optical"

    class BadAdapter:
        def infer(self, frame):
            return BadRaw()

    out = _tick(kernels, frame=_frame(), adapter=BadAdapter())
    assert out.mask is None
    assert out.decision.degraded is True


def test_n5_generic_exception_no_mask(kernels) -> None:
    spy = SpyAdapter(error=RuntimeError("gpu"))
    out = _tick(kernels, frame=_frame(), adapter=spy)
    assert out.mask is None
    assert out.decision.degraded is True


def test_n6_n7_collapse_from_gates_no_mask(kernels) -> None:
    spy = SpyAdapter(raw=_raw(score=0.1))
    out = _tick(kernels, frame=_frame(), adapter=spy)
    assert spy.calls == 1
    assert out.mask is None
    assert out.decision.degraded is True
    assert "min_known_fraction" not in Path(__file__).resolve().parents[1].joinpath(
        "compose", "tick.py"
    ).read_text()


def test_n8_n10_n12_happy_path(kernels) -> None:
    spy = SpyAdapter()
    out = _tick(kernels, frame=_frame(), adapter=spy)
    assert out.decision.publish_mask is True
    assert out.mask is not None
    assert out.mask.valid is True
    assert out.mask.age_s >= 0.0
    assert out.mask.scale == 1.0
    assert out.decision.degraded is False


def test_n8_n11_stale_no_mask(kernels) -> None:
    spy = SpyAdapter()
    stale_now = _STAMP + int(0.6 * _NS)
    out = _tick(kernels, frame=_frame(), adapter=spy, now_ns=stale_now)
    assert out.mask is None
    assert out.decision.publish_mask is False
    assert out.decision.degraded is True


def test_n9_frame_none(kernels) -> None:
    spy = SpyAdapter()
    out = _tick(kernels, frame=None, adapter=spy)
    assert spy.calls == 0
    assert out.mask is None
    assert out.decision.degraded is True
    assert out.decision.publish_mask is False


def test_n13_runner_up_none_in_tick() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("compose", "tick.py").read_text()
    assert "runner_up=None" in text
    assert "runner_up =" not in text.replace("runner_up=None", "")


def test_n16_no_adapter_id_on_port_meta() -> None:
    msg = Path(__file__).resolve().parents[1] / "port" / "port_meta.msg"
    fields = [
        line.split("#", 1)[0].strip()
        for line in msg.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    assert fields == [
        "std_msgs/Header header",
        "bool valid",
        "float32 age",
        "float32 scale",
    ]


def test_n17_stale_does_not_infer(kernels) -> None:
    spy = SpyAdapter()
    stale_now = _STAMP + int(0.6 * _NS)
    _tick(kernels, frame=_frame(), adapter=spy, now_ns=stale_now)
    assert spy.calls == 0


def test_n17_future_stamp_does_not_infer(kernels) -> None:
    spy = SpyAdapter()
    out = _tick(kernels, frame=_frame(stamp=_NOW + _NS), adapter=spy, now_ns=_NOW)
    assert spy.calls == 0
    assert out.mask is None
    assert out.decision.degraded is True


def test_post_infer_now_after_can_go_stale(kernels) -> None:
    spy = SpyAdapter()
    after = _STAMP + int(0.6 * _NS)
    out = _tick(kernels, frame=_frame(), adapter=spy, now_ns=_NOW, now_ns_after=after)
    assert spy.calls == 1
    assert out.mask is None
    assert out.decision.degraded is True


def test_live_ros_node_is_perception_cycle() -> None:
    from ugv_perception.node.cycle import perception_cycle
    from ugv_perception.ingest.msgs import ImageView, CameraInfoView

    spy = SpyAdapter()
    table, gates, fresh = (
        load_remap(_REMAP),
        load_gates(_GATES),
        load_freshness(_FRESH),
    )
    data = bytes([10, 20, 30] * 16)
    image = ImageView(
        stamp_ns=_STAMP,
        frame_id="camera_optical",
        height=4,
        width=4,
        encoding="rgb8",
        step=12,
        data=data,
    )
    info = CameraInfoView(
        stamp_ns=_STAMP,
        frame_id="camera_optical",
        height=4,
        width=4,
        k=(400.0, 0.0, 2.0, 0.0, 400.0, 2.0, 0.0, 0.0, 1.0),
    )
    out = perception_cycle(
        image=image,
        camera_info=info,
        now_ns=_NOW,
        adapter=spy,
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    assert spy.calls == 1
    assert out.mask is not None
    assert out.mask.header.stamp_ns == _STAMP
