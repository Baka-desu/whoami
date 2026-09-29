# WHOAMI perception workbench

Black + red web UI for the UGV perception stack. Upload or capture a photo, or take a live
camera / ROS 2 feed, and see the 3-class mask, depth, the safe path and a top-down ground map.

```
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + production build
```

## Status

- **Analysis is mock.** `src/analysis/mock.ts` produces the same shapes the real pipeline will
  (mask in `{0,1,2}`, metric depth, costmap grid, path, freshness). Replace that file when Dev 1's
  RUGD SegFormer + Depth Anything 3 outputs are available.
- **Sources:** photo upload, browser camera (start live detect / take photo), ROS 2 via
  rosbridge (`sensor_msgs/CompressedImage` + `CameraInfo`). The ROS 2 source is untested against a
  running rosbridge.
- Uploads and webcam frames have no `CameraInfo`, so the UI flags `K ASSUMED` and depth / ground
  map are not metric-valid.
