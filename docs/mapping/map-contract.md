# Map wire contract (gateway <-> web UI)

Copied from the plan (`docs/superpowers/plans/2026-10-02-3d-mapping-elevation.md`, Binary format v1) and the JSON contract used while building. Golden files both test suites share: `ugv_nav/ugv_api/test/fixtures/map/*.bin`. Code wins on any disagreement (`ugv_nav/ugv_api/ugv_api/mapcodec.py`, `ui/src/map/codec.ts`).

## Binary format v1

Little-endian. Decoders reject an unknown `format` and any length that does not match exactly.

```
prelude (24 bytes, all layers)
 0 char[4] magic   "UGVC" cloud | "UGVE" elevation | "UGVT" trajectory | "UGVG" cost grid
 4 u16 format = 1          6 u16 header_bytes
 8 u32 epoch (random per gateway process)      12 u32 seq       16 f64 stamp_s

UGVC  header 64: 24 u32 count | 28 u32 source_count | 32 f32 spacing_m | 36 u32 flags (bit0 = has rgb)
                 40 f32[3] bbox_min | 52 f32[3] bbox_max
      body: f32 xyz[3*count], then u8 rgb[3*count] if bit0          = 64 + (12 or 15) * count
UGVE  header 48: 24 u32 width (+x) | 28 u32 height (+y) | 32 f32 resolution_m | 36 f32 origin_x | 40 f32 origin_y
                 44 u32 known_cells
      body: f32 height[w*h] row-major (NaN = unknown), u8 obstacle[w*h] (5 cm units, 0 = none, saturates at 255),
            u8 confidence[w*h] (0..255)                             = 48 + 6 * w * h
UGVT  header 32: 24 u32 count | 28 f32 length_m
      body: f32[7*count]  x,y,z,qx,qy,qz,qw                         = 32 + 28 * count
UGVG  header 48: 24 u32 width | 28 u32 height | 32 f32 resolution_m | 36 f32 origin_x | 40 f32 origin_y | 44 f32 origin_yaw
      body: i8[w*h] row-major (-1 unknown, 0..100)                  = 48 + w * h
UGVD  header 40: 24 u32 width | 28 u32 height | 32 f32 unit_m (metres per count, 0.001) | 36 f32 max_range_m
      body: u16[w*h] row-major, 0 = hole, saturates at 65535        = 40 + 2 * w * h
```

Two image layers feed the RGB and depth panels of the map view:
- `depth` — `UGVD`, the DA3 depth image decimated by `map.depth_stride` (default 2, so 320x240).
- `camera` — the JPEG bytes of `/image_raw/compressed` passed through unchanged as `image/jpeg`. It has no prelude; its version is the `seq` in `MapStatus`.


## JSON contract for the map status and pose (gateway <-> UI)

The gateway's pydantic models use camelCase JSON (`alias_generator=to_camel`, `extra="forbid"`), like every
existing resource. Both sides implement exactly this.

## MapStatus — `GET /api/v1/map` and SSE event `map`

```json
{
  "epoch": 123456789,
  "seq": { "cloud": 0, "elevation": 0, "trajectory": 0, "grid": 0, "live": 0, "depth": 0, "camera": 0 },
  "stats": { "keyframes": 12, "depth_hz": 3.2, "calibration_placeholder": false }
}
```
- `epoch`: uint32, random per gateway process. `seq[layer]`: uint32, 0 = nothing received yet.
- All seven layer keys are always present.
- `stats`: a flat object. Keys are the snake_case names published by the ROS stats nodes, passed through
  unchanged (they are data keys, not model fields, so they are NOT camelCased). Values are number, string,
  boolean or null. It may be empty `{}`. Known keys: keyframes, loop_closures, path_length_m, db_bytes,
  last_update_age_s, mode, calibration_placeholder, mask_hz, depth_hz, depth_errors, cloud_source_points,
  elevation_known_cells, map_inputs_alive, map_rejects, map_restarts, map_last_reject. Unknown keys must be
  tolerated by the UI.
- The last four are the gateway's own health keys for its map inputs (not published by any ROS stats node):
  - `map_inputs_alive` (boolean): false when the map input thread is gone, has not ticked for 3 s, was never
    started, or is off because of a map-only configuration error; derived by the gateway, not by that thread.
  - `map_rejects` (integer): messages and frames refused or skipped since start (a layer in another frame, a
    malformed cloud, an unpaired or unstamped elevation half, a live frame without a transform, ...).
  - `map_restarts` (integer): times the supervisor restarted the map executor after an exception.
  - `map_last_reject` (string or null, at most 160 characters): `"<kind>: <reason>"` of the latest reject, or why
    the map inputs are off; null before the first one.

## Pose — SSE event `pose` (and `GET /api/v1/map/pose`)

```json
{ "available": true, "x": 1.0, "y": 2.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "ageS": 0.05 }
```
- `map -> base_link` from TF. When no transform has been seen: `"available": false` and every other field `null`
  (same pattern as the existing `BaseCommand` resource).

## Binary layer endpoints

`GET /api/v1/map/{cloud|elevation|trajectory|grid|live|depth}` -> `application/octet-stream` (Binary format v1).
`GET /api/v1/map/camera` -> `image/jpeg`.
Before a layer has data: HTTP 503 with an `application/problem+json` body.
