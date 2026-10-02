import { describe, expect, it } from 'vitest'
import { TH, TW, assumedIntrinsics, type Analysis, type FrameMeta } from '../types'
import { findPath } from './groundmap'
import { PathHold, type RobotPose } from './pathhold'

const pose = (x = 0, y = 0): RobotPose => ({ x, y, qx: 0, qy: 0, qz: 0, qw: 1 })

const meta = (stamp: number): FrameMeta => ({
  source: 'ros2', frameId: 'camera', stamp, receivedAt: stamp,
  width: 640, height: 480, K: assumedIntrinsics(640, 480), kAssumed: false, streaming: true,
})

function analysis(grid: Uint8Array, stamp: number, reasons: string[] = []): Analysis {
  return {
    meta: meta(stamp), mask: new Uint8Array(0), depth: null, grid,
    path: findPath(grid), pathPx: [], classPct: [0, 100, 0], depthStats: null,
    ageMs: 0, latencyMs: 0, degraded: reasons.length > 0, reasons,
  }
}

const open = () => new Uint8Array(TW * TH)
const lethal = () => new Uint8Array(TW * TH).fill(2)

describe('PathHold', () => {
  it('keeps the path while the pose has not moved', () => {
    const hold = new PathHold()
    hold.setPose(pose())
    const first = hold.apply(analysis(open(), 1))
    expect(first.path.length).toBeGreaterThan(1)
    const next = hold.apply(analysis(lethal(), 2))
    expect(next.path).toEqual(first.path)
  })

  it('drops the path after the corridor stays lethal', () => {
    const hold = new PathHold()
    hold.setPose(pose())
    const first = hold.apply(analysis(open(), 1))
    expect(hold.apply(analysis(lethal(), 2)).path).toEqual(first.path) // not lethal yet
    expect(hold.apply(analysis(lethal(), 3)).path).toEqual(first.path) // lethal, one frame
    expect(hold.apply(analysis(lethal(), 4)).path).toEqual([])
  })

  it('eases onto a new path after the pose moves, instead of snapping', () => {
    const hold = new PathHold()
    hold.setPose(pose())
    const first = hold.apply(analysis(open(), 1))
    const shifted = open()
    for (let z = 0; z < TH; z++) for (let x = (TW - 1) / 2; x < TW; x++) shifted[z * TW + x] = 2
    hold.setPose(pose(1, 0))
    hold.apply(analysis(shifted, 2)) // hazard not confirmed yet
    const eased = hold.apply(analysis(shifted, 3))
    const jumped = findPath(shifted)
    expect(eased.path.length).toBeGreaterThan(1)
    expect(eased.path).not.toEqual(first.path)
    expect(eased.path[eased.path.length - 1].x).not.toBeCloseTo(jumped[jumped.length - 1].x, 5)
  })

  it('lands on the new path once the post-move ease finishes', () => {
    const hold = new PathHold()
    hold.setPose(pose())
    hold.apply(analysis(open(), 1))
    const shifted = open()
    for (let z = 0; z < TH; z++) for (let x = (TW - 1) / 2; x < TW; x++) shifted[z * TW + x] = 2
    hold.setPose(pose(1, 0))
    let last = hold.apply(analysis(shifted, 2))
    for (let stamp = 3; stamp <= 8; stamp++) last = hold.apply(analysis(shifted, stamp))
    expect(last.path).toEqual(findPath(shifted))
  })

  it('keeps the debounced grid when the same mask is drawn again', () => {
    const hold = new PathHold()
    const first = hold.apply(analysis(lethal(), 1))
    expect(first.grid[0]).toBe(1) // one hazard frame is not lethal yet
    const again = hold.apply(analysis(lethal(), 1))
    expect(again.grid[0]).toBe(1)
    expect(again.grid).not.toBe(first.grid)
  })

  it('with no pose, ignores one far goal and eases on the next', () => {
    const hold = new PathHold()
    const first = hold.apply(analysis(open(), 1))
    const shifted = open()
    for (let z = 0; z < TH; z++) for (let x = (TW - 1) / 2; x < TW; x++) shifted[z * TW + x] = 2
    expect(hold.apply(analysis(shifted, 2)).path).toEqual(first.path) // hazard not confirmed yet
    expect(hold.apply(analysis(shifted, 3)).path).toEqual(first.path) // one far goal is ignored
    const eased = hold.apply(analysis(shifted, 4))
    expect(eased.path).not.toEqual(first.path)
    expect(eased.path.length).toBeGreaterThan(1)
  })

  it('does not count the same mask twice', () => {
    const hold = new PathHold()
    hold.setPose(pose())
    const first = hold.apply(analysis(open(), 1))
    expect(hold.apply(analysis(lethal(), 1)).path).toEqual(first.path)
    expect(hold.apply(analysis(lethal(), 2)).path).toEqual(first.path)
  })

  it('forgets a held path when the mask is invalid', () => {
    const hold = new PathHold()
    hold.setPose(pose())
    hold.apply(analysis(open(), 1))
    const bad = analysis(lethal(), 2, ['INVALID MASK'])
    expect(hold.apply(bad).path).toEqual(bad.path)
    const again = hold.apply(analysis(open(), 3))
    expect(again.path.length).toBeGreaterThan(1)
  })
})
