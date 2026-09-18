"""CanonicalMask → sensor_msgs Image. PortMeta fields until the .msg is compiled."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, MultiArrayLayout

from ugv_perception.compose.tick import ComposeOut
from ugv_perception.port.ids import CONF_ENCODING, MASK_ENCODING
from ugv_perception.port.mask import CanonicalMask

_NS = 1_000_000_000


def _fill_stamp(msg_stamp: object, stamp_ns: int) -> None:
    msg_stamp.sec = stamp_ns // _NS
    msg_stamp.nanosec = stamp_ns % _NS


def mask_to_image_msg(mask: CanonicalMask) -> Image:
    h, w = mask.classes.shape
    msg = Image()
    _fill_stamp(msg.header.stamp, mask.header.stamp_ns)
    msg.header.frame_id = mask.header.frame_id
    msg.height = h
    msg.width = w
    msg.encoding = MASK_ENCODING
    msg.is_bigendian = False
    msg.step = w
    msg.data = bytes(np.ascontiguousarray(mask.classes).reshape(-1))
    return msg


def confidence_to_image_msg(mask: CanonicalMask) -> Image:
    h, w = mask.confidence.shape
    msg = Image()
    _fill_stamp(msg.header.stamp, mask.header.stamp_ns)
    msg.header.frame_id = mask.header.frame_id
    msg.height = h
    msg.width = w
    msg.encoding = CONF_ENCODING
    msg.is_bigendian = False
    msg.step = w * 4
    msg.data = bytes(np.ascontiguousarray(mask.confidence).tobytes())
    return msg


@dataclass(frozen=True, slots=True)
class WireOut:
    degraded: Bool
    mask: Image | None
    confidence: Image | None
    port_meta: Float64MultiArray | None


def wire_compose_out(out: ComposeOut) -> WireOut:
    degraded = Bool(data=out.decision.degraded)
    if out.mask is None or not out.decision.publish_mask:
        return WireOut(degraded=degraded, mask=None, confidence=None, port_meta=None)
    mask_msg = mask_to_image_msg(out.mask)
    conf_msg = confidence_to_image_msg(out.mask)
    meta = Float64MultiArray()
    meta.layout = MultiArrayLayout(
        dim=[
            MultiArrayDimension(label="valid", size=1, stride=3),
            MultiArrayDimension(label="age", size=1, stride=2),
            MultiArrayDimension(label="scale", size=1, stride=1),
        ],
        data_offset=0,
    )
    meta.data = [
        1.0 if out.mask.valid else 0.0,
        float(out.mask.age_s),
        float(out.mask.scale),
    ]
    return WireOut(
        degraded=degraded,
        mask=mask_msg,
        confidence=conf_msg,
        port_meta=meta,
    )
