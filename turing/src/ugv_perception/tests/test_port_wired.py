"""T10 wired port contract. compose_tick + wire_compose_out. No camera, no IR.

Mandatory file: no rclpy spin. sensor_msgs Image types come from wire_compose_out
(the consumer layout). Missing ROS packages fail collection, not skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.adapter.output import AdapterError
from ugv_perception.compose import compose_tick, load_compose_configs
from ugv_perception.node.wire import wire_compose_out
from ugv_perception.port.ids import CANONICAL, CONF_ENCODING, MASK_ENCODING
from ugv_perception.port.mask import make_mask
from ugv_perception.tests.fixtures import OMIT, FixtureAdapter, FixtureSource

_ROOT = Path(__file__).resolve().parents[3]
_REMAP = _ROOT / "config" / "ontologies" / "yoloe.yaml"
_GATES = _ROOT / "config" / "perception" / "yoloe.yaml"
_FRESH = _ROOT / "config" / "perception" / "port.yaml"

_NS = 1_000_000_000
_STAMP = 2_000_000_000
_NOW_OK = _STAMP + 10_000_000  # T+10ms
_NOW_STALE = _STAMP + 600_000_000  # T+0.6s vs max 0.50s
_BANNED = ("dirt_path", "openvino_gpu", "adapter_id", "yoloe-26s", "/cmd_vel")


@pytest.fixture
def kernels():
    return load_compose_configs(
        remap_path=_REMAP,
        gates_path=_GATES,
        freshness_path=_FRESH,
    )


def _source_frame(stamp: int = _STAMP, frame_id: str = "camera_optical"):
    return FixtureSource().frame(stamp_ns=stamp, frame_id=frame_id, hw=(2, 2))


def _wire(kernels, *, frame, adapter, now_ns: int = _NOW_OK):
    table, gates, fresh = kernels
    out = compose_tick(
        frame=frame,
        now_ns=now_ns,
        adapter=adapter,
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    return wire_compose_out(out)


def _fail_closed(wired) -> None:
    assert wired.degraded.data is True
    assert wired.mask is None
    assert wired.confidence is None
    assert wired.port_meta is None


def _mask_pixels(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w)


def _conf_pixels(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    return np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(h, w)


def _stamp_of(msg) -> int:
    return int(msg.header.stamp.sec) * _NS + int(msg.header.stamp.nanosec)


def test_w1_w2_mixed_canonical_mono8(kernels) -> None:
    labels = np.array([[1, 4], [5, 1]], dtype=np.int32)
    adapter = FixtureAdapter(label_ids=labels)
    frame = _source_frame()
    wired = _wire(kernels, frame=frame, adapter=adapter)
    assert wired.mask is not None
    assert wired.mask.encoding == MASK_ENCODING
    pixels = _mask_pixels(wired.mask)
    assert pixels.dtype == np.uint8
    assert set(int(x) for x in np.unique(pixels).tolist()) <= CANONICAL
    assert pixels.tolist() == [[1, 0], [2, 1]]


def test_w3_stamp_is_sensor_not_now(kernels) -> None:
    frame = _source_frame()
    wired = _wire(kernels, frame=frame, adapter=FixtureAdapter(), now_ns=_NOW_OK)
    assert wired.mask is not None
    assert _stamp_of(wired.mask) == _STAMP
    assert _stamp_of(wired.mask) != _NOW_OK


def test_w4_frame_id_copied(kernels) -> None:
    frame = _source_frame(frame_id="camera_optical")
    wired = _wire(kernels, frame=frame, adapter=FixtureAdapter())
    assert wired.mask is not None
    assert wired.mask.header.frame_id == "camera_optical"


def test_w5_confidence_aligned(kernels) -> None:
    wired = _wire(kernels, frame=_source_frame(), adapter=FixtureAdapter())
    assert wired.mask is not None
    assert wired.confidence is not None
    assert wired.confidence.encoding == CONF_ENCODING
    assert _stamp_of(wired.confidence) == _stamp_of(wired.mask)
    assert wired.confidence.header.frame_id == wired.mask.header.frame_id
    assert (wired.confidence.height, wired.confidence.width) == (
        wired.mask.height,
        wired.mask.width,
    )
    conf = _conf_pixels(wired.confidence)
    assert conf.dtype == np.float32
    assert bool(np.isfinite(conf).all())
    assert float(conf.min()) >= 0.0
    assert float(conf.max()) <= 1.0


def test_w6_port_meta_stopgap_not_health(kernels) -> None:
    wired = _wire(kernels, frame=_source_frame(), adapter=FixtureAdapter())
    assert wired.mask is not None
    assert wired.port_meta is not None
    assert len(wired.port_meta.data) == 3
    assert wired.port_meta.data[0] == 1.0
    assert abs(float(wired.port_meta.data[1]) - 0.01) < 1e-9
    assert wired.port_meta.data[2] == 1.0
    labels = [d.label for d in wired.port_meta.layout.dim]
    assert "adapter_id" not in labels
    assert "degraded" not in labels
    assert labels == ["valid", "age", "scale"]


def test_w7_stale_from_now_minus_stamp(kernels) -> None:
    frame = _source_frame()
    wired = _wire(
        kernels, frame=frame, adapter=FixtureAdapter(), now_ns=_NOW_STALE
    )
    _fail_closed(wired)


def test_w8_stale_does_not_restamp_prior_mask(kernels) -> None:
    frame = _source_frame()
    adapter = FixtureAdapter()
    first = _wire(kernels, frame=frame, adapter=adapter, now_ns=_NOW_OK)
    assert first.mask is not None
    first_stamp = _stamp_of(first.mask)
    second = _wire(kernels, frame=frame, adapter=adapter, now_ns=_NOW_STALE)
    _fail_closed(second)
    assert _stamp_of(first.mask) == first_stamp == _STAMP


def test_wire_adapter_exception_degrades(kernels) -> None:
    wired = _wire(
        kernels,
        frame=_source_frame(),
        adapter=FixtureAdapter(error=AdapterError("boom")),
    )
    _fail_closed(wired)


def test_wire_adapter_none_degrades(kernels) -> None:
    wired = _wire(
        kernels,
        frame=_source_frame(),
        adapter=FixtureAdapter(return_none=True),
    )
    _fail_closed(wired)


def test_wire_adapter_missing_stamp_degrades(kernels) -> None:
    wired = _wire(
        kernels,
        frame=_source_frame(),
        adapter=FixtureAdapter(stamp_ns=OMIT),
    )
    _fail_closed(wired)


def test_wire_adapter_stamp_mismatch_degrades(kernels) -> None:
    wired = _wire(
        kernels,
        frame=_source_frame(),
        adapter=FixtureAdapter(stamp_ns=_STAMP + 1),
    )
    _fail_closed(wired)


def test_wire_adapter_missing_frame_degrades(kernels) -> None:
    wired = _wire(
        kernels,
        frame=_source_frame(),
        adapter=FixtureAdapter(frame_id=OMIT),
    )
    _fail_closed(wired)


def test_wire_adapter_frame_mismatch_degrades(kernels) -> None:
    wired = _wire(
        kernels,
        frame=_source_frame(),
        adapter=FixtureAdapter(frame_id="other_optical"),
    )
    _fail_closed(wired)


def test_w10_low_score_publishes_unknown_not_traversable(kernels) -> None:
    scores = np.full((2, 2), 0.1, dtype=np.float32)
    wired = _wire(
        kernels,
        frame=_source_frame(),
        adapter=FixtureAdapter(raw_scores=scores),
    )
    assert wired.degraded.data is False
    assert wired.mask is not None
    assert wired.confidence is not None
    assert wired.port_meta is not None
    pix = _mask_pixels(wired.mask)
    assert set(int(x) for x in np.unique(pix).tolist()) == {0}


def test_w11_missing_remap_prevents_mask(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_compose_configs(
            remap_path=tmp_path / "missing.yaml",
            gates_path=_GATES,
            freshness_path=_FRESH,
        )


def test_w12_no_model_leak_on_wire(kernels) -> None:
    labels = np.array([[1, 4], [5, 1]], dtype=np.int32)
    wired = _wire(
        kernels, frame=_source_frame(), adapter=FixtureAdapter(label_ids=labels)
    )
    assert wired.mask is not None
    blob = " ".join(
        [
            wired.mask.encoding,
            wired.mask.header.frame_id,
            wired.confidence.encoding if wired.confidence is not None else "",
            " ".join(d.label for d in wired.port_meta.layout.dim)
            if wired.port_meta is not None
            else "",
        ]
    )
    for word in _BANNED:
        assert word not in blob


def test_w13_illegal_pixel_aborts_and_never_published(kernels) -> None:
    with pytest.raises(ValueError):
        make_mask(
            stamp_ns=_STAMP,
            frame_id="camera_optical",
            classes=np.array([[3, 0], [1, 2]], dtype=np.uint8),
            confidence=np.full((2, 2), 0.9, dtype=np.float32),
            producer_ok=True,
            age_s=0.01,
            scale=1.0,
            source_hw=(2, 2),
        )
    labels = np.array([[1, 4], [5, 1]], dtype=np.int32)
    wired = _wire(
        kernels, frame=_source_frame(), adapter=FixtureAdapter(label_ids=labels)
    )
    assert wired.mask is not None
    assert int(_mask_pixels(wired.mask).max()) <= 2


def test_w14_degraded_bool_when_no_mask(kernels) -> None:
    wired = _wire(
        kernels, frame=_source_frame(), adapter=FixtureAdapter(), now_ns=_NOW_STALE
    )
    assert type(wired.degraded.data) is bool
    _fail_closed(wired)


def test_w15_fixtures_not_in_production() -> None:
    root = Path(__file__).resolve().parents[1]
    for folder in ("adapter", "ingest", "compose", "node"):
        for py in (root / folder).glob("*.py"):
            text = py.read_text(encoding="utf-8")
            assert "FixtureSource" not in text
            assert "DummySource" not in text
            assert "LiveCameraSource" not in text


def test_w16_t10_sources_clean() -> None:
    here = Path(__file__).resolve().parent
    paths = [
        here / "test_port_wired.py",
        here / "test_port_wired_ros.py",
        here / "fixtures" / "source.py",
        here / "fixtures" / "adapter.py",
        here / "fixtures" / "__init__.py",
    ]
    for py in paths:
        text = py.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "openvino" not in stripped
                assert "DepthFrame" not in stripped
                if py.name == "test_port_wired.py":
                    assert "rclpy" not in stripped
