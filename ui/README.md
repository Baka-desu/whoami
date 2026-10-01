# UGV operator console (Dev 5)

One web app. The **camera view is the main page**: the live camera with Dev 1's Perception Port drawn over it
(class mask, metric depth, flat-ground path preview). The operator controls and status sit in two sidebars.

| Where | What | Source |
|---|---|---|
| Centre | Camera + mask / depth / path overlay, freshness banners, layer toggles | rosbridge (read only) |
| Left sidebar | E-stop (§3.1 level 1) | `PUT /api/v1/safety/e-stop` |
| Left sidebar | Mapping / localize mode (§10) | `PUT /api/v1/localization/mode` |
| Left sidebar | Map-frame NavigateToPose goal (§10, §11) | `POST /api/v1/navigation/goals`, `DELETE …/{id}` |
| Left sidebar | Camera source: robot camera (default), browser camera, photo upload | rosbridge / browser |
| Right sidebar | §12 health table + safety arbiter status | `GET /api/v1/safety/status` |
| Right sidebar | Final `/cmd_vel` (read only), navigation, localization, e-stop state | `GET /api/v1/base/command`, telemetry |
| Right sidebar | Perception: ground map preview, class shares, depth, health, source/K | rosbridge (read only) |

Gateway values arrive over `GET /api/v1/telemetry/stream` (Server-Sent Events). `architecture.md` §14 defers a
UI for v1 (the v1 operator interface is the CLI); this console is an approved extension.

```
npm install
npm run dev      # http://localhost:5173, proxies /api to the gateway (UGV_API_URL, default http://127.0.0.1:8080)
npm run build    # typecheck + production build
npm test         # unit tests (vitest)
npm run lint
```

Start the gateway with `ros2 launch ugv_api api.launch.py` (package `ugv_nav/ugv_api`). The camera view needs a
`rosbridge_server` on port 9090 of the page's host and Dev 5's camera driver (`/image_raw/compressed` +
`/camera_info`).

## Rules it keeps

- **Commands only through Dev 5's gateway.** E-stop, mode and goals go to `/api/v1`. rosbridge is confined to the
  read-only camera view (`CameraView`, `Viewport`, `Inspector`, `SourcePanel`, `source/rosbridge.ts`,
  `source/useCameraSource.ts`, `analysis/`). A test (`src/source/api.test.ts`) fails if any other file touches
  rosbridge or Dev 1/2/4 topics.
- **Never commands motion.** Nothing publishes to ROS: no `/cmd_vel`, no `/cmd_vel_nav2`, no advertise or service
  calls. The gateway can't publish drive commands either.
- **Fail closed.** Every payload is validated; a summary that contradicts its rows is not trusted; once the
  stream is older than 2 s everything reads NO SIGNAL and goals are disabled. The gateway re-checks the goal
  gate and answers `409` with the tripped watches. A mask that isn't `{0,1,2}` becomes all-unknown and is flagged
  `INVALID MASK`; a mask older than 500 ms shows STALE. Robot frames without CameraInfo are not drawn.
- **Not the safety authority.** The §12 table is the operator's view. Dev 5's arbiter (`ugv_safety`) is what
  zeros `/cmd_vel`; the board says NOT RUNNING while `/ugv/safety_status` has no publisher.
- **Path is a preview.** The path overlay and the ground map are a flat-ground approximation from the mask. They are
  not Dev 3's costmap and not Nav2's plan.
- **Browser camera and photo uploads** have no perception backend, so they show `NO ANALYZER` and `K ASSUMED`;
  nothing is faked. The mock in `analysis/__fixtures__/` is imported only by tests.
