# Perception workbench (Dev 1 eval tool)

Offline / bring-up viewer for Dev 1's Perception Port (`architecture.md` §4 `bag` / `rugd` profiles). It is
**not** part of the product's operator interface: the operator console is `ui/` and talks only to Dev 5's
gateway (`ugv_nav/ugv_api`). This tool was split out of `ui/` so the operator UI never reads Dev 1 directly.

```
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + production build
npm test         # unit tests (vitest)
npm run lint
```

## What it shows

- **ROS 2 source (rosbridge, read only).** Camera (`sensor_msgs/CompressedImage` on
  `/camera/image_raw/compressed` + `/camera/camera_info`, i.e. Dev 5's camera contract republished by
  `image_transport`) and Dev 1's port: `/segmentation/mask`, `/segmentation/port_meta`,
  `/ugv/perception_degraded`, and `/perception/depth/image` when the DA3 weights are exported. Each mask is
  checked against the contract (`source/rosimage.ts`); a mask that is not `{0,1,2}` becomes an all-unknown
  mask flagged `INVALID MASK`. Each analysis carries the mask's own age, so a stalled perception node goes
  STALE even while camera frames keep arriving.
- **Ground map / path preview** (`analysis/groundmap.ts`) is a flat-ground approximation from the mask for
  eyeballing a bag. It is not Dev 3's costmap and not Nav2's plan.
- **Uploads and the browser camera** have no perception backend, so they stay at `NO ANALYZER` with
  `K ASSUMED`; nothing is faked. The mock in `analysis/__fixtures__/` is imported only by tests.

It publishes nothing: no goals, no e-stop, no `/cmd_vel*`.

Needs a running `rosbridge_server` (not launched by any package yet) and, for the camera, Dev 5's driver.
