"""Camera driver helpers (no ROS, no OpenCV): calibration -> CameraInfo fields, capture-size
check, device parsing and the rate pacing of the UI stream.

The calibration itself is loaded and validated by Dev 2's ugv_localization.camera (the single place
that refuses zero / fake K), so a driver can never publish a lying CameraInfo.
"""

from __future__ import annotations

from dataclasses import dataclass


class CaptureError(RuntimeError):
    """The camera cannot produce frames that match its calibration."""


@dataclass(frozen=True, slots=True)
class CameraInfoFields:
    width: int
    height: int
    distortion_model: str
    d: tuple[float, ...]
    k: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]


def camera_info_fields(cal: object) -> CameraInfoFields:
    """Fields of a sensor_msgs/CameraInfo from a validated ugv_localization CameraCalibration."""
    return CameraInfoFields(
        width=cal.width,  # type: ignore[attr-defined]
        height=cal.height,  # type: ignore[attr-defined]
        distortion_model=cal.distortion_model,  # type: ignore[attr-defined]
        d=tuple(cal.d),  # type: ignore[attr-defined]
        k=tuple(cal.k),  # type: ignore[attr-defined]
        r=tuple(cal.r),  # type: ignore[attr-defined]
        p=tuple(cal.p),  # type: ignore[attr-defined]
    )


def check_capture_size(cal_width: int, cal_height: int, got_width: int, got_height: int) -> None:
    """K is only valid at the resolution it was calibrated at: refuse a camera that delivers another."""
    if (got_width, got_height) != (cal_width, cal_height):
        raise CaptureError(
            f"camera delivers {got_width}x{got_height} but the calibration is for {cal_width}x{cal_height}; "
            "recalibrate at this resolution or set the camera to the calibrated one"
        )


def parse_device(value: str) -> int | str:
    """`0` becomes index 0; `/dev/video0` (or a URL / file) stays as given."""
    v = str(value).strip()
    if not v:
        raise CaptureError("device parameter is empty")
    return int(v) if v.isdigit() else v


class RatePacer:
    """True at most `rate_hz` times per second, given calls at a higher, steady frame rate.

    `slack_s` (about half a frame period) stops a 30 fps camera landing just short of every deadline.
    """

    def __init__(self, rate_hz: float, slack_s: float = 0.0) -> None:
        if not rate_hz > 0.0:
            raise ValueError("rate_hz must be > 0")
        self._period = 1.0 / rate_hz
        self._slack = max(0.0, slack_s)
        self._last: float | None = None

    def due(self, now_s: float) -> bool:
        if self._last is None or now_s - self._last >= self._period - self._slack:
            self._last = now_s
            return True
        return False
