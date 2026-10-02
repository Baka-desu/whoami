import { describe, expect, it } from 'vitest'
import type { RosDepth, RosMask } from '../source/rosimage'
import { GH, GW, assumedIntrinsics, type FrameMeta } from '../types'
import { RosPerception } from './ros-analyzer'

const meta = (over: Partial<FrameMeta> = {}): FrameMeta => ({
  source: 'ros2', frameId: 'camera_optical_frame', stamp: Date.now(), receivedAt: Date.now(),
  width: 640, height: 480, K: assumedIntrinsics(640, 480), kAssumed: false, streaming: true, ...over,
})

// lower part of the image traversable, a hazard patch ahead on the left, the rest unknown
function mask(over: Partial<RosMask> = {}): RosMask {
  const w = 64, h = 48
  const data = new Uint8Array(w * h)
  for (let y = 30; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = 1
  for (let y = 36; y < 44; y++) for (let x = 4; x < 12; x++) data[y * w + x] = 2
  return { stampMs: 1000, frameId: 'camera_optical_frame', width: w, height: h, data, receivedAt: Date.now(), ...over }
}

const depth = (over: Partial<RosDepth> = {}): RosDepth => ({
  stampMs: 1000, frameId: 'camera_optical_frame', width: 4, height: 4, data: new Float32Array(16).fill(5), receivedAt: Date.now(), ...over,
})

const fakeFrame = {} as ImageBitmap

describe('RosPerception', () => {
  it('is unavailable and returns nothing until a mask arrives', async () => {
    const p = new RosPerception()
    expect(p.available).toBe(false)
    expect(await p.analyze(fakeFrame, meta())).toBeNull()
  })

  it('turns a fresh Perception Port mask into an Analysis with a legal {0,1,2} mask and a path', async () => {
    const p = new RosPerception()
    p.pushMask(mask())
    expect(p.available).toBe(true)
    const a = (await p.analyze(fakeFrame, meta()))!
    expect(a.mask).toHaveLength(GW * GH)
    expect([...new Set(a.mask)].every((c) => c <= 2)).toBe(true)
    expect(a.path.length).toBeGreaterThan(1)
    expect(a.reasons).toEqual([])
    expect(a.degraded).toBe(false)
    expect(a.depth).toBeNull() // no depth channel received
    expect(a.depthStats).toBeNull()
    expect(a.mock).toBeUndefined()
  })

  it('goes stale on the MASK age even though camera frames keep arriving', async () => {
    const p = new RosPerception()
    const old = Date.now() - 3000
    p.pushMask(mask({ receivedAt: old }))
    const a = (await p.analyze(fakeFrame, meta({ receivedAt: Date.now() })))!
    expect(a.reasons).toContain('MASK STALE')
    expect(a.degraded).toBe(true)
    expect(a.meta.receivedAt).toBe(old) // freshness reads the mask, not the camera frame
  })

  it('fails safe on a mask that is not {0,1,2}: all unknown, flagged invalid', async () => {
    const p = new RosPerception()
    const bad = mask()
    bad.data[5] = 3
    p.pushMask(bad)
    const a = (await p.analyze(fakeFrame, meta()))!
    expect(a.reasons).toContain('INVALID MASK')
    expect(a.classPct).toEqual([100, 0, 0])
    expect(a.path).toEqual([])
  })

  it('flags a frame_id mismatch between mask and image', async () => {
    const p = new RosPerception()
    p.pushMask(mask({ frameId: 'other_camera' }))
    expect((await p.analyze(fakeFrame, meta()))!.reasons).toContain('FRAME MISMATCH')
  })

  it('ignores a leading slash when comparing frame ids', async () => {
    const p = new RosPerception()
    p.pushMask(mask({ frameId: '/camera_optical_frame' }))
    expect((await p.analyze(fakeFrame, meta()))!.reasons).not.toContain('FRAME MISMATCH')
  })

  it('reports Dev 1 degraded and port-invalid flags', async () => {
    const p = new RosPerception()
    p.pushMask(mask())
    p.setHealth({ degraded: true })
    p.setHealth({ valid: false })
    const a = (await p.analyze(fakeFrame, meta()))!
    expect(a.reasons).toEqual(expect.arrayContaining(['PERCEPTION DEGRADED', 'PERCEPTION PORT INVALID']))
    expect(a.degraded).toBe(true)
  })

  it('uses depth only when it belongs to the same image stamp', async () => {
    const p = new RosPerception()
    p.pushMask(mask({ stampMs: 1000 }))
    p.pushDepth(depth({ stampMs: 999 }))
    expect((await p.analyze(fakeFrame, meta()))!.depth).toBeNull()
    p.pushDepth(depth({ stampMs: 1000 }))
    const a = (await p.analyze(fakeFrame, meta()))!
    expect(a.depth).toHaveLength(GW * GH)
    expect(a.depthStats).toEqual({ min: 5, median: 5, max: 5 })
  })

  it('never lets an older mask replace a newer one', async () => {
    const p = new RosPerception()
    p.pushMask(mask({ stampMs: 2000 }))
    p.pushMask(mask({ stampMs: 1000 }))
    expect((await p.analyze(fakeFrame, meta()))!.meta.stamp).toBe(2000)
  })

  it('keeps the drawn path while the robot pose has not moved', async () => {
    const p = new RosPerception()
    p.setPose({ x: 0, y: 0, qx: 0, qy: 0, qz: 0, qw: 1 })
    p.pushMask(mask({ stampMs: 1000 }))
    const first = (await p.analyze(fakeFrame, meta()))!
    const blocked = mask({ stampMs: 2000 })
    blocked.data.fill(2)
    p.pushMask(blocked)
    const held = (await p.analyze(fakeFrame, meta()))!
    expect(held.path).toEqual(first.path)
  })

  it('forgets everything on reset (a reconnect must not reuse an old mask)', async () => {
    const p = new RosPerception()
    p.pushMask(mask())
    p.setHealth({ degraded: true })
    p.reset()
    expect(p.available).toBe(false)
    expect(await p.analyze(fakeFrame, meta())).toBeNull()
  })
})
