"""T11 live p95. Skipped until Dev 5 publishes a real Image+CameraInfo stream."""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="IR on disk; outdoor Image+CameraInfo still Dev 5 — no dummy outdoor RGB, no fake FPS"
)
def test_llive_p95_capture_to_mask() -> None:
    raise AssertionError("must stay skipped")
