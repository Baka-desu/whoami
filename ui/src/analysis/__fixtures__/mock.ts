// TEST FIXTURE ONLY - never import this from production code (App, components, analyzer.ts).
// A fake perception result with the Analysis shape, for tests. A test in analysis.test.ts fails
// if any non-test module imports it.
//   mask  : colour distance from the ground patch straight ahead -> {0,1,2}
//   depth : flat-ground model from camera height + intrinsics
//   grid / path / stats: the production geometry in ../groundmap.ts (this file only fakes the mask + depth)
import { CAM_H, GH, GW, type Analysis, type FrameMeta } from '../../types'
import { buildAnalysis, cameraGrid } from '../groundmap'

const FAR_M = 30

const scratch = document.createElement('canvas')
scratch.width = GW
scratch.height = GH
const sctx = scratch.getContext('2d', { willReadFrequently: true })!

export function analyze(frame: CanvasImageSource, meta: FrameMeta): Analysis {
  const t0 = performance.now()
  sctx.drawImage(frame, 0, 0, GW, GH)
  const px = sctx.getImageData(0, 0, GW, GH).data

  const { fy, vh } = cameraGrid(meta)

  const mask = segment(px, vh)
  const depth = new Float32Array(GW * GH)
  for (let v = 0; v < GH; v++) {
    const d = v > vh ? Math.min(FAR_M, (CAM_H * fy) / (v - vh)) : FAR_M
    depth.fill(d, v * GW, (v + 1) * GW)
  }

  return { ...buildAnalysis({ meta, mask, depth, latencyMs: performance.now() - t0 }), mock: true }
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
