// TEST FIXTURE ONLY - never import this from production code (App, components, analyzer.ts).
// A fake perception result with the Analysis shape, for tests. A test in analysis.test.ts fails
// if any non-test module imports it.
//   mask  : colour distance from the ground patch straight ahead -> {0,1,2}
//   depth : flat-ground model from camera height + intrinsics
//   grid  : ground-plane costmap sampled back out of the mask (hazard wins, unknown != free)
//   path  : Dijkstra from the robot to the farthest free cell
import {
  CAM_H, CELL_M, GH, GW, PERCEPTION_MAX_AGE_MS, TH, TW, Z_MIN,
  type Analysis, type FrameMeta,
} from '../../types'

const HORIZON = Math.round(GH * 0.42)
const FAR_M = 30

const scratch = document.createElement('canvas')
scratch.width = GW
scratch.height = GH
const sctx = scratch.getContext('2d', { willReadFrequently: true })!

export function analyze(frame: CanvasImageSource, meta: FrameMeta): Analysis {
  const t0 = performance.now()
  sctx.drawImage(frame, 0, 0, GW, GH)
  const px = sctx.getImageData(0, 0, GW, GH).data

  const fx = (meta.K.fx * GW) / meta.width
  const fy = (meta.K.fy * GH) / meta.height
  const cx = (meta.K.cx * GW) / meta.width
  const vh = meta.kAssumed ? HORIZON : Math.round((meta.K.cy * GH) / meta.height)

  const mask = segment(px, vh)
  const depth = new Float32Array(GW * GH)
  for (let v = 0; v < GH; v++) {
    const d = v > vh ? Math.min(FAR_M, (CAM_H * fy) / (v - vh)) : FAR_M
    depth.fill(d, v * GW, (v + 1) * GW)
  }

  const grid = groundGrid(mask, fx, fy, cx, vh)
  const path = findPath(grid)
  const pathPx = path.map(({ x, z }) => ({
    u: (cx + (fx * x) / z) / GW,
    v: (vh + (CAM_H * fy) / z) / GH,
  }))

  const counts = [0, 0, 0]
  for (const c of mask) counts[c]++
  const classPct = counts.map((n) => (100 * n) / mask.length) as [number, number, number]

  const ground = depth.slice(vh + 1 < GH ? (vh + 1) * GW : 0).sort()
  const depthStats = {
    min: ground[0],
    median: ground[Math.floor(ground.length / 2)],
    max: ground[ground.length - 1],
  }

  const ageMs = Math.max(0, Date.now() - (meta.receivedAt ?? meta.stamp))
  const reasons: string[] = []
  if (ageMs > PERCEPTION_MAX_AGE_MS) reasons.push('MASK STALE')
  if (classPct[1] < 3) reasons.push('NO TRAVERSABLE GROUND')

  return {
    meta, mask, depth, grid, path, pathPx, classPct, depthStats, ageMs,
    latencyMs: performance.now() - t0,
    degraded: reasons.length > 0,
    reasons,
    mock: true,
  }
}

function segment(px: Uint8ClampedArray, vh: number): Uint8Array {
  let r = 0, g = 0, b = 0, n = 0
  for (let v = GH - 14; v < GH; v++) {
    for (let u = GW / 2 - 20; u < GW / 2 + 20; u++) {
      const i = (v * GW + u) * 4
      r += px[i]; g += px[i + 1]; b += px[i + 2]; n++
    }
  }
  r /= n; g /= n; b /= n

  const raw = new Uint8Array(GW * GH)
  for (let v = vh + 1; v < GH; v++) {
    for (let u = 0; u < GW; u++) {
      const i = (v * GW + u) * 4
      const dist = Math.hypot(px[i] - r, px[i + 1] - g, px[i + 2] - b) / 441.7
      raw[v * GW + u] = dist < 0.12 ? 1 : dist < 0.25 ? 0 : 2
    }
  }

  // 3x3 majority filter to drop speckle
  const out = raw.slice()
  for (let v = 1; v < GH - 1; v++) {
    for (let u = 1; u < GW - 1; u++) {
      const votes = [0, 0, 0]
      for (let dv = -1; dv <= 1; dv++) for (let du = -1; du <= 1; du++) votes[raw[(v + dv) * GW + u + du]]++
      const best = votes.indexOf(Math.max(...votes))
      if (votes[best] >= 5) out[v * GW + u] = best
    }
  }
  return out
}

// Flat-ground projection only holds where something touches the floor. Keep the bottom
// CONTACT_PX rows of each hazard run in a column and mark the rest SKIP, so a tall trunk
// or rock is not smeared across the ground map.
const CONTACT_PX = 2
const SKIP = 255

function contactMask(mask: Uint8Array): Uint8Array {
  const out = mask.slice()
  for (let u = 0; u < GW; u++) {
    let run = 0
    for (let v = GH - 1; v >= 0; v--) {
      const i = v * GW + u
      run = mask[i] === 2 ? run + 1 : 0
      if (run > CONTACT_PX) out[i] = SKIP
    }
  }
  return out
}

// Sample the mask into a ground grid by projecting each cell back into the image.
function groundGrid(rawMask: Uint8Array, fx: number, fy: number, cx: number, vh: number): Uint8Array {
  const mask = contactMask(rawMask)
  const grid = new Uint8Array(TW * TH)
  for (let gz = 0; gz < TH; gz++) {
    const z = Z_MIN + gz * CELL_M
    const v = vh + (CAM_H * fy) / z
    const dv = Math.abs((CAM_H * fy) / z - (CAM_H * fy) / (z + CELL_M))
    const wv = Math.max(1, Math.round(dv / 2))
    const wu = Math.max(1, Math.round((fx * CELL_M) / z / 2))
    for (let gx = 0; gx < TW; gx++) {
      const x = (gx - (TW - 1) / 2) * CELL_M
      const u = cx + (fx * x) / z
      let trav = 0, haz = 0, total = 0
      for (let vv = Math.round(v - wv); vv <= Math.round(v + wv); vv++) {
        for (let uu = Math.round(u - wu); uu <= Math.round(u + wu); uu++) {
          if (uu < 0 || uu >= GW || vv <= vh || vv >= GH) continue
          const c = mask[vv * GW + uu]
          if (c === SKIP) continue
          if (c === 1) trav++
          else if (c === 2) haz++
          total++
        }
      }
      // unseen or unsure cells stay 1 (inflated): unknown is never free
      grid[gz * TW + gx] = total === 0 ? 1 : haz / total > 0.15 ? 2 : trav / total > 0.5 ? 0 : 1
    }
  }

  const inflated = grid.slice()
  for (let gz = 0; gz < TH; gz++) {
    for (let gx = 0; gx < TW; gx++) {
      if (grid[gz * TW + gx] !== 2) continue
      for (let dz = -2; dz <= 2; dz++) {
        for (let dx = -2; dx <= 2; dx++) {
          const z = gz + dz, x = gx + dx
          if (z < 0 || z >= TH || x < 0 || x >= TW) continue
          const near = Math.max(Math.abs(dz), Math.abs(dx)) <= 1
          const i = z * TW + x
          inflated[i] = Math.max(inflated[i], near ? 2 : 1)
        }
      }
    }
  }
  return inflated
}

function findPath(grid: Uint8Array): { x: number; z: number }[] {
  const start = (TW - 1) / 2
  const dist = new Float32Array(TW * TH).fill(Infinity)
  const prev = new Int32Array(TW * TH).fill(-1)
  const done = new Uint8Array(TW * TH)
  dist[start] = 0

  for (;;) {
    let cur = -1
    for (let i = 0; i < dist.length; i++) if (!done[i] && dist[i] < Infinity && (cur < 0 || dist[i] < dist[cur])) cur = i
    if (cur < 0) break
    done[cur] = 1
    const cz = Math.floor(cur / TW), cxg = cur % TW
    for (let dz = -1; dz <= 1; dz++) {
      for (let dx = -1; dx <= 1; dx++) {
        const z = cz + dz, x = cxg + dx
        if ((dz === 0 && dx === 0) || z < 0 || z >= TH || x < 0 || x >= TW) continue
        const ni = z * TW + x
        if (grid[ni] === 2) continue
        const step = (dz !== 0 && dx !== 0 ? 1.414 : 1) * (grid[ni] === 0 ? 1 : 6)
        if (dist[cur] + step < dist[ni]) { dist[ni] = dist[cur] + step; prev[ni] = cur }
      }
    }
  }

  let goal = start, best = -Infinity
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] !== 0 || dist[i] === Infinity) continue
    const score = Math.floor(i / TW) - 0.08 * dist[i]
    if (score > best) { best = score; goal = i }
  }
  if (goal === start) return []

  const cells: number[] = []
  for (let i = goal; i >= 0; i = prev[i]) cells.push(i)
  cells.reverse()
  const pts = cells.map((i) => ({ x: ((i % TW) - (TW - 1) / 2) * CELL_M, z: Z_MIN + Math.floor(i / TW) * CELL_M }))
  return pts.map((p, i) => {
    if (i === 0 || i === pts.length - 1) return p
    return { x: (pts[i - 1].x + p.x + pts[i + 1].x) / 3, z: p.z }
  })
}
