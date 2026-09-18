"""Source-frame identity for every port message (architecture §8.4)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrameHeader:
    stamp_ns: int
    frame_id: str

    def __post_init__(self) -> None:
        if type(self.stamp_ns) is not int:
            raise TypeError(
                f"stamp_ns must be a Python int, got {type(self.stamp_ns).__name__}"
            )
        if self.stamp_ns <= 0:
            raise ValueError(f"stamp_ns must be > 0, got {self.stamp_ns}")
        if type(self.frame_id) is not str:
            raise TypeError(
                f"frame_id must be a Python str, got {type(self.frame_id).__name__}"
            )
        if self.frame_id == "":
            raise ValueError("frame_id must be a non-empty optical frame name")
