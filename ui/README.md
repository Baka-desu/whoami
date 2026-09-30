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
- **Robot / navigation (ROS 2 source).** Over rosbridge the UI reads Dev 2 (`ugv_nav`:
  `/ugv/pose_valid`, `/ugv/localization_status`, `/ugv/localization/odom_source`, `/odom`, `/tf`)
  and Dev 4 (`ugv_navigation`: `/ugv/nav2_heartbeat`, `/ugv/nav2_status`, `/cmd_vel_nav2`, `/plan`,
  `/local_costmap/costmap`), draws the Nav2 costmap + plan, and can send / cancel a
  `/navigate_to_pose` goal. A goal is relative to a start pose frozen from Dev 2's TF (archV1.md
  §9) and is blocked unless the pose is valid, Nav2 is heartbeating, perception is not degraded
  and e-stop is not asserted. A heartbeat that stops reads NO SIGNAL (fail-closed). `E-STOP`
  publishes `/ugv/e_stop` (Dev 5, spec only - no code yet) and is re-published until released;
  the UI never publishes `/cmd_vel*`. `/ugv/safety_status` is not shown: its type is undefined.
- **Sources:** photo upload, browser camera (start live detect / take photo), ROS 2 via
  rosbridge (`sensor_msgs/CompressedImage` + `CameraInfo`). The ROS 2 source is untested against a
  running rosbridge.
- Uploads and webcam frames have no `CameraInfo`, so the UI flags `K ASSUMED` and depth / ground
  map are not metric-valid.
