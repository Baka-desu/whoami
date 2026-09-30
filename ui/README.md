# WHOAMI perception workbench

Black + red web UI for the UGV perception stack. Upload or capture a photo, or take a live
camera / ROS 2 feed, and see the 3-class mask, depth, the safe path and a top-down ground map.

```
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + production build
npm test         # unit tests (vitest)
```

## Status

- **Analysis goes through one seam.** `src/analysis/analyzer.ts` defines the `Analyzer` interface
  that `App.tsx` calls. No real perception backend is connected yet, so the shipped analyzer is
  `unavailableAnalyzer`: frames are shown, the top bar reads `NO ANALYZER`, and there is no mask,
  depth or path. Dev 1's REST analyzer plugs in by implementing `Analyzer` (returning the
  Perception Port result: mask in `{0,1,2}`, metric depth, freshness) - no UI change needed.
- **Mock analysis is a test fixture only.** `src/analysis/__fixtures__/mock.ts` is imported by
  tests, never by production code (a test fails if it is). It is not a stand-in to replace later.
- **Sources:** photo upload, browser camera (start live detect / take photo), ROS 2 via
  rosbridge (`sensor_msgs/CompressedImage` + `CameraInfo`). The ROS 2 source is untested against a
  running rosbridge.
- Uploads and webcam frames have no `CameraInfo`, so the UI flags `K ASSUMED` and depth / ground
  map are not metric-valid.
