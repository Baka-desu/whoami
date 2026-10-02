// The production analyzer for the ROS 2 source: Dev 1's Perception Port outputs (mask, depth, health), received over
// rosbridge, turned into an Analysis. Nothing is computed from the camera image here; the mask is Dev 1's.
//
// The UI frame and the mask are different messages, so each analysis is paired with the NEWEST mask and reports
// the mask's own age: a perception node that stalls goes STALE even while camera frames keep arriving (the
// returned meta carries the mask's stamp and arrival time, which is what the freshness clock reads).
// The drawn path is then held (pathhold.ts): it stays put until the gateway pose moves, and it is dropped
// when its corridor stays lethal.
import { GH, GW, PERCEPTION_MAX_AGE_MS, type Analysis, type FrameMeta } from '../types'
import type { RosDepth, RosMask } from '../source/rosimage'
import type { Analyzer } from './analyzer'
import { buildAnalysis } from './groundmap'
import { PathHold, type RobotPose } from './pathhold'

export interface PerceptionHealth {
  degraded?: boolean // /ugv/perception_degraded
  valid?: boolean // /segmentation/port_meta valid flag
}

// How long after the last mask the perception node still counts as connected (STALE shows well before this).
const CONNECTED_MS = 5000

const frameKey = (f: string) => f.replace(/^\//, '')

function resampleMask(m: RosMask): Uint8Array | null {
  for (let i = 0; i < m.data.length; i++) if (m.data[i] > 2) return null // not a {0,1,2} Perception Port mask
  const out = new Uint8Array(GW * GH)
  for (let v = 0; v < GH; v++) {
    const sy = Math.min(m.height - 1, Math.floor((v * m.height) / GH))
    for (let u = 0; u < GW; u++) out[v * GW + u] = m.data[sy * m.width + Math.min(m.width - 1, Math.floor((u * m.width) / GW))]
  }
  return out
}

function resampleDepth(d: RosDepth): Float32Array {
  const out = new Float32Array(GW * GH)
  for (let v = 0; v < GH; v++) {
    const sy = Math.min(d.height - 1, Math.floor((v * d.height) / GH))
    for (let u = 0; u < GW; u++) out[v * GW + u] = d.data[sy * d.width + Math.min(d.width - 1, Math.floor((u * d.width) / GW))]
  }
  return out
}

export class RosPerception implements Analyzer {
  private mask: RosMask | null = null
  private depth: RosDepth | null = null
  private health: PerceptionHealth = {}
  private path = new PathHold()

  get available(): boolean {
    return this.mask !== null && Date.now() - this.mask.receivedAt <= CONNECTED_MS
  }

  pushMask(m: RosMask): void {
    if (this.mask && m.stampMs < this.mask.stampMs) return // never let an older mask replace a newer one
    this.mask = m
  }

  pushDepth(d: RosDepth): void {
    if (this.depth && d.stampMs < this.depth.stampMs) return
    this.depth = d
  }

  setHealth(h: PerceptionHealth): void {
    this.health = { ...this.health, ...h }
  }

  // Gateway pose (map -> base_link). Null when telemetry is down: the path then eases instead of freezing.
  setPose(pose: RobotPose | null): void {
    this.path.setPose(pose)
  }

  reset(): void {
    this.mask = null
    this.depth = null
    this.health = {}
    this.path.reset()
  }

  async analyze(_frame: ImageBitmap, meta: FrameMeta): Promise<Analysis | null> {
    const mask = this.mask
    if (!mask) return null
    const t0 = performance.now()
    const reasons: string[] = []

    if (Date.now() - mask.receivedAt > PERCEPTION_MAX_AGE_MS) reasons.push('MASK STALE')
    if (this.health.degraded) reasons.push('PERCEPTION DEGRADED')
    if (this.health.valid === false) reasons.push('PERCEPTION PORT INVALID')
    if (mask.frameId && meta.frameId && frameKey(mask.frameId) !== frameKey(meta.frameId)) reasons.push('FRAME MISMATCH')

    // A contract violation fails safe: an all-unknown mask (unknown is never free), never a guess.
    let classes = resampleMask(mask)
    if (!classes) {
      classes = new Uint8Array(GW * GH)
      reasons.push('INVALID MASK')
    }

    // Depth is published after its mask with the same image stamp; use it only when it belongs to this mask.
    const d = this.depth
    const depth = d && d.stampMs === mask.stampMs && frameKey(d.frameId) === frameKey(mask.frameId) ? resampleDepth(d) : null

    return this.path.apply(buildAnalysis({
      meta: { ...meta, stamp: mask.stampMs, receivedAt: mask.receivedAt },
      mask: classes, depth, latencyMs: performance.now() - t0, reasons,
    }))
  }
}
