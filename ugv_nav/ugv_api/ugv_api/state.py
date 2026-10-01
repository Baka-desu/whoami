"""Thread-safe cache of the latest sample per watched input. Pure Python, no ROS.

Written by the ROS executor thread, read by HTTP handlers. Every sample keeps the time the gateway
received it and, when the message has a header, the source stamp (architecture §8.4: age = now - stamp).
All times are nanoseconds on the gateway node's clock (sim time aware).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

# Keys: one per input the gateway watches (all are Dev 5 subscriptions per dev.md §3, or Dev 5's own).
CAMERA_INFO = "camera_info"  # /camera/camera_info, stamp only (§12 camera watch)
MASK = "mask"  # /segmentation/mask, stamp only (§12 perception watch)
PERCEPTION_DEGRADED = "perception_degraded"  # /ugv/perception_degraded
POSE_VALID = "pose_valid"  # /ugv/pose_valid
LOCALIZATION_STATUS = "localization_status"  # /ugv/localization_status
TF_MAP_BASE = "tf_map_base"  # TF map->base_link (value = None, stamp = transform stamp)
NAV2_HEARTBEAT = "nav2_heartbeat"  # /ugv/nav2_heartbeat
NAV2_STATUS = "nav2_status"  # /ugv/nav2_status
E_STOP = "e_stop"  # /ugv/e_stop as observed on the graph (any publisher)
SAFETY_STATUS = "safety_status"  # /ugv/safety_status (Dev 5 arbiter, not implemented yet)
CMD_VEL = "cmd_vel"  # /cmd_vel, the final command (read only)


@dataclass(frozen=True)
class Sample:
    value: Any
    received_ns: int
    stamp_ns: int | None = None

    def age_s(self, now_ns: int, *, use_stamp: bool = False) -> float:
        """Seconds since the source stamp (use_stamp and a stamp exists) or since receipt."""
        ref = self.stamp_ns if use_stamp and self.stamp_ns else self.received_ns
        return (now_ns - ref) / 1e9


class StateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, Sample] = {}

    def put(self, key: str, value: Any, received_ns: int, stamp_ns: int | None = None) -> None:
        with self._lock:
            self._samples[key] = Sample(value, received_ns, stamp_ns)

    def get(self, key: str) -> Sample | None:
        with self._lock:
            return self._samples.get(key)

    def seen(self) -> list[str]:
        with self._lock:
            return sorted(self._samples)
