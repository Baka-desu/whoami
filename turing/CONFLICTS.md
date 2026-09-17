# architecture.md over dev.md

`dev.md` is a work-split. It is not a second architecture. Where they disagree, we follow `architecture.md`.

| `dev.md` wording | Architecture | What we do |
|---|---|---|
| Tutorial ONNX listed in the same YOLOE production task | ONNX is **adapter scaffold / eval only**. Kill list: tutorial-as-only-ontology | T09 is last, not default, not required to ship `live_cam` |
| Depth Anything in the perception pipeline with YOLOE | Geometry **side-channel**, not the port. §9 | T08 publishes depth/cloud only. Never writes `/segmentation/mask` |
| Port described as mask + degraded only | Port = mask + **conf** + **freshness** + **frame** + **valid** (§8) | T01/T07 publish confidence + `port_meta`, not just mask |
| “Front ROI lethal” implied as Dev 1 | §8.6 is a system fail-safe; costmaps are Dev 3 | We publish degraded; we do not write costmap cells |
| `source:=outdoor_path.mp4` as the run example | `live_cam` is the product default (§4) | Video/bag is an eval source behind the same `Source` interface |
| Hour boxes (29h) and difficulty ranks | Hours live outside architecture; ~30h/person is a soft aim | Ignore as design constraints |
| “Hardware-agnostic inference abstraction” as a headline | Stack is YOLOE + ONNX + optional Depth Anything | Keep a thin `InferenceBackend` seam (T06) — good modularity, not a new product |
| Independent test on RUGD as first-class | RUGD is an eval profile | Allowed as eval if you provide the dataset; not a product source |

Nothing else in the Dev 1 `dev.md` section is in conflict. Sensor ingestion, remap, confidence, freshness, YOLOE outdoor adapter all stand.
