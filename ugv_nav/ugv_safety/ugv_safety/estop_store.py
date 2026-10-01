"""Persist the latched e-stop across an arbiter restart (no ROS).

A latched (transient-local) message lives only as long as the publisher that sent it, and the UI's
publisher disappears when the UI disconnects. So after "operator asserts e-stop, UI closes, arbiter
restarts" nothing on the bus remembers the kill. The arbiter therefore keeps its own copy on disk and
starts from it.

Fail safe: a missing file means the e-stop was never asserted (first start). A file that exists but cannot
be read, or says anything other than `0`, counts as asserted, so a corrupted record never lets the robot move.
"""

from __future__ import annotations

import os
from pathlib import Path


class EstopLatchStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(os.path.expanduser(str(path)))

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> tuple[bool, str]:
        """(asserted, note). `note` explains a state restored from disk, or why it was assumed asserted."""
        try:
            text = self._path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return False, ""
        except (OSError, UnicodeDecodeError) as exc:
            return True, f"cannot read e-stop record {self._path} ({exc}); assuming asserted"
        if text == "0":
            return False, ""
        if text == "1":
            return True, f"e-stop was asserted before shutdown ({self._path}); staying asserted until an explicit release"
        return True, f"e-stop record {self._path} is corrupt ({text[:20]!r}); assuming asserted"

    def save(self, asserted: bool) -> bool:
        """Write atomically. False if it could not be persisted (the in-memory latch still works)."""
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="ascii") as fh:
                fh.write("1\n" if asserted else "0\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
            return True
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            return False
