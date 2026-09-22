"""Scripted Adapter for T10. Copies stamp/frame unless a test overrides."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import ADAPTER_ID, UNLABELED_NAME, AdapterError, RawSemOutput

COPY = object()
OMIT = object()

_DEFAULT_NAMES = {0: UNLABELED_NAME, 1: "dirt_path", 4: "sky", 5: "person"}


class FixtureAdapter:
    def __init__(
        self,
        *,
        label_ids: np.ndarray | None = None,
        raw_scores: np.ndarray | None = None,
        id_to_name: dict[int, str] | None = None,
        error: BaseException | None = None,
        return_none: bool = False,
        stamp_ns: object = COPY,
        frame_id: object = COPY,
    ) -> None:
        self.calls = 0
        self._label_ids = label_ids
        self._raw_scores = raw_scores
        self._id_to_name = id_to_name if id_to_name is not None else dict(_DEFAULT_NAMES)
        self._error = error
        self._return_none = return_none
        self._stamp_ns = stamp_ns
        self._frame_id = frame_id

    def infer(self, frame: ImageFrame) -> RawSemOutput | None:
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._return_none:
            return None
        hw = (int(frame.rgb.shape[0]), int(frame.rgb.shape[1]))
        labels = self._label_ids
        if labels is None:
            labels = np.ones(hw, dtype=np.int32)
        scores = self._raw_scores
        if scores is None:
            scores = np.full(hw, 0.9, dtype=np.float32)
        if self._stamp_ns is OMIT or self._frame_id is OMIT:
            ns: dict[str, object] = {
                "adapter_id": ADAPTER_ID,
                "label_ids": labels,
                "raw_scores": scores,
                "id_to_name": self._id_to_name,
                "hw": hw,
            }
            if self._stamp_ns is not OMIT:
                ns["stamp_ns"] = frame.stamp_ns if self._stamp_ns is COPY else self._stamp_ns
            if self._frame_id is not OMIT:
                ns["frame_id"] = frame.frame_id if self._frame_id is COPY else self._frame_id
            return SimpleNamespace(**ns)  # type: ignore[return-value]
        stamp = frame.stamp_ns if self._stamp_ns is COPY else self._stamp_ns
        fid = frame.frame_id if self._frame_id is COPY else self._frame_id
        return RawSemOutput(
            adapter_id=ADAPTER_ID,
            label_ids=np.asarray(labels, dtype=np.int32),
            raw_scores=np.asarray(scores, dtype=np.float32),
            id_to_name=self._id_to_name,
            stamp_ns=stamp,  # type: ignore[arg-type]
            frame_id=fid,  # type: ignore[arg-type]
            hw=hw,
        )
