"""One perception cycle: optional decode → compose_tick. No rclpy."""

from __future__ import annotations

from ugv_perception.compose.tick import ComposeOut, compose_tick
from ugv_perception.confidence.profile import GateProfile
from ugv_perception.freshness.profile import FreshnessProfile
from ugv_perception.ingest.decode import decode_frame
from ugv_perception.ingest.msgs import CameraInfoView, ImageView
from ugv_perception.remap.table import RemapTable


def perception_cycle(
    *,
    image: ImageView | None,
    camera_info: CameraInfoView | None,
    now_ns: int,
    adapter: object,
    remap_table: RemapTable,
    gate_profile: GateProfile,
    freshness_profile: FreshnessProfile,
    now_ns_after: int | None = None,
) -> ComposeOut:
    frame = None
    if image is not None and camera_info is not None:
        try:
            frame = decode_frame(image, camera_info)
        except (TypeError, ValueError):
            frame = None
    return compose_tick(
        frame=frame,
        now_ns=now_ns,
        adapter=adapter,
        remap_table=remap_table,
        gate_profile=gate_profile,
        freshness_profile=freshness_profile,
        now_ns_after=now_ns_after,
    )
