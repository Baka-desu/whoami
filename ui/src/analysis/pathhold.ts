// The camera path is a fresh search on every mask (groundmap.ts). Mask noise moves that line even when the
// robot has not moved. This holds the line the UI draws:
//   - while the gateway pose is still, keep the path;
//   - a corridor that stays lethal is dropped, so a new obstacle is not ignored;
//   - once the pose moves, ease onto the new path over a few masks instead of snapping;
//   - with no pose, keep a close goal and ease only after a far goal persists.
// The mask overlay is untouched. buildAnalysis stays a pure function of one mask.
import { CELL_M, TH, TW, Z_MIN, type Analysis } from '../types'
import { findPath, pathPixels } from './groundmap'

const MOVE_M = 0.08
const MOVE_YAW = (5 * Math.PI) / 180
const HAZARD_FRAMES = 2
const BLOCK_FRAMES = 2
const FAR_FRAMES = 2
const SETTLE_FRAMES = 6
const EASE = 0.35
const GOAL_CLOSE_M = 0.75

export interface RobotPose {
  x: number
  y: number
  qx: number
  qy: number
  qz: number
  qw: number
}

interface Pt { x: number; z: number }

// Heading about +z. Same ZYX formula as map/scene-math yawOf.
function yawOf(qx: number, qy: number, qz: number, qw: number): number {
  return Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
}

function angDiff(a: number, b: number): number {
  return Math.atan2(Math.sin(a - b), Math.cos(a - b))
}

function pathBlocked(path: Pt[], grid: Uint8Array): boolean {
  for (const p of path) {
    const gx = Math.round(p.x / CELL_M + (TW - 1) / 2)
    const gz = Math.round((p.z - Z_MIN) / CELL_M)
    if (gx < 0 || gx >= TW || gz < 0 || gz >= TH) continue
    if (grid[gz * TW + gx] === 2) return true
  }
  return false
}

function goalsClose(a: Pt[], b: Pt[]): boolean {
  if (a.length === 0 || b.length === 0) return false
  const p = a[a.length - 1]
  const q = b[b.length - 1]
  return Math.hypot(p.x - q.x, p.z - q.z) < GOAL_CLOSE_M
}

function resample(path: Pt[], n: number): Pt[] {
  if (path.length === 0 || n <= 0) return []
  if (path.length === 1 || n === 1) return Array.from({ length: n }, () => ({ ...path[0] }))
  const out: Pt[] = []
  for (let i = 0; i < n; i++) {
    const f = (i / (n - 1)) * (path.length - 1)
    const i0 = Math.floor(f)
    const i1 = Math.min(path.length - 1, i0 + 1)
    const u = f - i0
    out.push({
      x: path[i0].x + (path[i1].x - path[i0].x) * u,
      z: path[i0].z + (path[i1].z - path[i0].z) * u,
    })
  }
  return out
}

function ease(prev: Pt[], next: Pt[]): Pt[] {
  if (next.length === 0) return prev.map((p) => ({ ...p }))
  if (prev.length === 0) return next.map((p) => ({ ...p }))
  const n = Math.max(prev.length, next.length)
  const a = resample(prev, n)
  const b = resample(next, n)
  return a.map((p, i) => ({
    x: p.x + (b[i].x - p.x) * EASE,
    z: p.z + (b[i].z - p.z) * EASE,
  }))
}

export class PathHold {
  private on = new Uint8Array(0)
  private off = new Uint8Array(0)
  private stable = new Uint8Array(0)
  private shown: Pt[] | null = null
  private anchor: { x: number; y: number; yaw: number } | null = null
  private pose: { x: number; y: number; yaw: number } | null = null
  private blockStreak = 0
  private emptyStreak = 0
  private farStreak = 0
  private settling = 0
  private lastStamp: number | null = null

  setPose(pose: RobotPose | null): void {
    if (!pose || ![pose.x, pose.y, pose.qx, pose.qy, pose.qz, pose.qw].every(Number.isFinite)) {
      this.pose = null
      return
    }
    this.pose = { x: pose.x, y: pose.y, yaw: yawOf(pose.qx, pose.qy, pose.qz, pose.qw) }
  }

  reset(): void {
    this.on = new Uint8Array(0)
    this.off = new Uint8Array(0)
    this.stable = new Uint8Array(0)
    this.shown = null
    this.anchor = null
    this.blockStreak = 0
    this.emptyStreak = 0
    this.farStreak = 0
    this.settling = 0
    this.lastStamp = null
  }

  apply(raw: Analysis): Analysis {
    if (raw.reasons.includes('INVALID MASK')) {
      this.reset()
      return raw
    }
    // The same mask is analysed on every camera frame. Count a mask once.
    if (this.lastStamp === raw.meta.stamp && this.shown) {
      // Later camera frames of this mask must keep the debounced grid the path was searched on.
      const path = this.shown.map((p) => ({ ...p }))
      return { ...raw, grid: this.stable.slice(), path, pathPx: pathPixels(path, raw.meta) }
    }
    this.lastStamp = raw.meta.stamp

    const grid = this.stableGrid(raw.grid)
    const candidate = findPath(grid)
    const moved = this.pose !== null && this.shown !== null && this.moved()
    if (moved) this.settling = SETTLE_FRAMES

    const blocked = this.shown !== null && this.shown.length > 0 && pathBlocked(this.shown, grid)
    if (blocked) this.blockStreak++
    else this.blockStreak = 0
    // No route, and the held cells are not lethal either (inflated or unknown). Drop after the same wait.
    const routeGone = this.shown !== null && this.shown.length > 0 && candidate.length === 0 && !blocked
    if (routeGone) this.emptyStreak++
    else this.emptyStreak = 0

    let next: Pt[]
    if (this.blockStreak >= BLOCK_FRAMES || this.emptyStreak >= BLOCK_FRAMES) {
      // The corridor stayed lethal, or the search stayed empty. Take the new search, including an empty one.
      next = candidate
      this.blockStreak = 0
      this.emptyStreak = 0
      this.settling = 0
      this.farStreak = 0
    } else if (this.shown === null || this.shown.length === 0) {
      next = candidate
    } else if (candidate.length === 0) {
      next = this.shown
    } else if (this.pose && !moved && this.settling === 0) {
      next = this.shown
      this.farStreak = 0
    } else if (!this.pose && this.settling === 0 && goalsClose(this.shown, candidate)) {
      next = this.shown
      this.farStreak = 0
    } else if (!this.pose && this.settling === 0) {
      this.farStreak++
      if (this.farStreak < FAR_FRAMES) {
        next = this.shown
      } else {
        // A far goal that persists eases, then lands. One ease step used to stop inside GOAL_CLOSE_M.
        this.settling = SETTLE_FRAMES
        this.farStreak = 0
        next = ease(this.shown, candidate)
        this.settling--
      }
    } else {
      this.farStreak = 0
      next = ease(this.shown, candidate)
      if (this.settling > 0) this.settling--
      // The last step lands on the new path. Stopping at a 35% ease would freeze short of it.
      if (this.settling === 0) next = candidate
    }

    const changed = next !== this.shown
    this.shown = next
    if (changed && this.pose && (moved || this.anchor === null)) this.anchor = { ...this.pose }
    const path = next.map((p) => ({ ...p }))
    return { ...raw, grid, path, pathPx: pathPixels(path, raw.meta) }
  }

  private moved(): boolean {
    const pose = this.pose
    const anchor = this.anchor
    if (!pose || !anchor) return true
    const d = Math.hypot(pose.x - anchor.x, pose.y - anchor.y)
    return d >= MOVE_M || Math.abs(angDiff(pose.yaw, anchor.yaw)) >= MOVE_YAW
  }

  // A cell becomes lethal after HAZARD_FRAMES hazard observations and stays lethal for the same
  // count after it clears, so one flipped mask frame neither opens nor closes a corridor.
  private stableGrid(raw: Uint8Array): Uint8Array {
    if (this.on.length !== raw.length) {
      this.on = new Uint8Array(raw.length)
      this.off = new Uint8Array(raw.length)
      this.stable = new Uint8Array(raw.length)
    }
    const out = new Uint8Array(raw.length)
    for (let i = 0; i < raw.length; i++) {
      if (raw[i] === 2) {
        this.on[i] = Math.min(HAZARD_FRAMES, this.on[i] + 1)
        this.off[i] = 0
      } else {
        this.off[i] = Math.min(HAZARD_FRAMES, this.off[i] + 1)
        this.on[i] = 0
      }
      if (this.on[i] >= HAZARD_FRAMES) out[i] = 2
      else if (this.stable[i] === 2 && this.off[i] < HAZARD_FRAMES) out[i] = 2
      else out[i] = raw[i] === 2 ? 1 : raw[i]
      this.stable[i] = out[i]
    }
    return out
  }
}
