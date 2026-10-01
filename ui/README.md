# UGV operator console (Dev 5)

Web console for the five operator items `architecture.md` defines, and nothing else:

| Item | Architecture | Gateway resource |
|---|---|---|
| E-stop | §3.1 level 1 | `PUT /api/v1/safety/e-stop` |
| Health timeout table + safety arbiter status | §12, dev.md Dev 5 task 6 | `GET /api/v1/safety/status` |
| Final `/cmd_vel` (read only) | §3.1 | `GET /api/v1/base/command` |
| Mapping / localize mode | §10 | `PUT /api/v1/localization/mode` |
| Map-frame NavigateToPose goal | §10, §11 | `POST /api/v1/navigation/goals`, `DELETE …/{id}` |

Live values arrive over `GET /api/v1/telemetry/stream` (Server-Sent Events). `architecture.md` §14 defers a
UI for v1 (the v1 operator interface is the CLI); this console is an approved extension limited to the table above.

```
npm install
npm run dev      # http://localhost:5173, proxies /api to the gateway (UGV_API_URL, default http://127.0.0.1:8080)
npm run build    # typecheck + production build
npm test         # unit tests (vitest)
npm run lint
```

Start the gateway with `ros2 launch ugv_api api.launch.py` (package `ugv_nav/ugv_api`).

## Rules it keeps

- **Only Dev 5's gateway.** No rosbridge, no direct reads of Dev 1/2/4 topics, no camera, mask, depth,
  costmap or plan. A test (`src/source/api.test.ts`) fails if production code references any of them.
- **Never commands motion.** It cannot publish `/cmd_vel` or `/cmd_vel_nav2`; the gateway can't either.
- **Fail closed.** Every payload is validated; a summary that contradicts its rows is not trusted; once the
  stream is older than 2 s everything reads NO SIGNAL and goals are disabled. The gateway re-checks the goal
  gate and answers `409` with the tripped watches.
- **Not the safety authority.** The §12 table is the operator's view. Dev 5's arbiter (`ugv_safety`, not
  implemented yet) is what zeros `/cmd_vel`; the board says so while `/ugv/safety_status` has no publisher.

The perception workbench (photo / webcam / bag viewer of Dev 1's mask and depth) moved to
`turing/workbench/` as a Dev 1 eval tool.
