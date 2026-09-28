#!/usr/bin/env bash
# You run this. It does not run from tests and it does not wire the node.
# Depth Anything 3 Metric Large — Apache 2.0, monocular metric depth.
# https://huggingface.co/depth-anything/DA3METRIC-LARGE
# Meters later: depth_m = focal_px * raw / 300. Not downloaded by pytest.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/weights/da3metric-large"
mkdir -p "${DEST}"
python -m pip install --user 'huggingface_hub>=0.24'
DEST="${DEST}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

dest = os.environ["DEST"]
snapshot_download(repo_id="depth-anything/DA3METRIC-LARGE", local_dir=dest)
print("saved", dest)
print("Do not commit the safetensors. T08 is not wired until subarch8 is frozen.")
PY
