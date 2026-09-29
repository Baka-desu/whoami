# WHOAMI perception workbench

Black + red web UI for the UGV perception stack. Upload or capture a photo, or take a live
camera / ROS 2 feed, and see the 3-class mask, depth, the safe path and a top-down ground map.

```
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + production build
```

## Analysis: real backend, with a mock fallback

For real segmentation + real metric depth, start the backend in [`../server`](../server/README.md)
(one-time setup, then `uvicorn app:app --host 127.0.0.1 --port 8008`). The UI polls
`http://127.0.0.1:8008/health` every 5 seconds and automatically switches between **REAL
ANALYSIS** and **MOCK ANALYSIS** — no toggle, and no restart needed on either side.

With no backend running, `src/analysis/mock.ts` produces the same shapes real analysis does
(mask in `{0,1,2}`, metric depth, costmap grid, path, freshness), so the UI is fully usable
without any Python setup.

Read `../server/README.md` before assuming the backend matches `dev.md` exactly — it's real
model inference, but a stand-in for the specific RUGD SegFormer / Depth Anything 3 models named
there, and it cannot meet the architecture's 500ms live perception budget on modest hardware
(the UI will honestly show `STALE` for live sources when that happens; that's real latency, not
a bug).

- **Sources:** photo upload, browser camera (start live detect / take photo), ROS 2 via
  rosbridge (`sensor_msgs/CompressedImage` + `CameraInfo`). The ROS 2 source is untested against a
  running rosbridge — there's no ROS 2 install on this machine.
- Uploads and webcam frames have no `CameraInfo`, so the UI flags `K ASSUMED` and depth / ground
  map are not metric-valid.
