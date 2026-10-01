import { describe, expect, it } from 'vitest'
import { parseDepth, parseMask } from './rosimage'

const b64 = (bytes: Uint8Array) => btoa(String.fromCharCode(...bytes))
const header = { stamp: { sec: 10, nanosec: 500_000_000 }, frame_id: 'cam' }

const maskMsg = (over: Record<string, unknown> = {}) => ({
  header, width: 4, height: 2, encoding: 'mono8', step: 4, is_bigendian: 0,
  data: b64(Uint8Array.from([0, 1, 2, 1, 0, 1, 2, 1])), ...over,
})

describe('parseMask', () => {
  it('decodes a valid mono8 mask', () => {
    const m = parseMask(maskMsg(), 123)!
    expect(m.width).toBe(4)
    expect(Array.from(m.data)).toEqual([0, 1, 2, 1, 0, 1, 2, 1])
    expect(m.stampMs).toBe(10_500)
    expect(m.frameId).toBe('cam')
    expect(m.receivedAt).toBe(123)
  })

  it.each([
    ['wrong encoding', { encoding: 'rgb8' }],
    ['wrong step', { step: 8 }],
    ['short data', { data: b64(Uint8Array.from([0, 1])) }],
    ['bad base64', { data: '!!!not base64!!!' }],
    ['no stamp', { header: { frame_id: 'cam' } }],
    ['zero size', { width: 0 }],
    ['huge size', { width: 100000, height: 100000 }],
  ])('rejects %s', (_label, over) => {
    expect(parseMask(maskMsg(over), 0)).toBeNull()
  })

  it('accepts a plain number array as data', () => {
    expect(parseMask(maskMsg({ data: [0, 1, 2, 1, 0, 1, 2, 1] }), 0)).not.toBeNull()
  })

  it('rejects garbage input', () => {
    expect(parseMask(null, 0)).toBeNull()
    expect(parseMask('x', 0)).toBeNull()
  })
})

describe('parseDepth', () => {
  const floats = new Float32Array([1.5, 2, NaN, 30])
  const depthMsg = (over: Record<string, unknown> = {}) => ({
    header, width: 2, height: 2, encoding: '32FC1', step: 8, is_bigendian: 0,
    data: b64(new Uint8Array(floats.buffer)), ...over,
  })

  it('decodes little-endian float32 metres and keeps NaN holes', () => {
    const d = parseDepth(depthMsg(), 0)!
    expect(d.data[0]).toBe(1.5)
    expect(d.data[1]).toBe(2)
    expect(Number.isNaN(d.data[2])).toBe(true)
    expect(d.data[3]).toBe(30)
  })

  it.each([
    ['wrong encoding', { encoding: '16UC1' }],
    ['big endian', { is_bigendian: 1 }],
    ['wrong step', { step: 4 }],
    ['short data', { data: b64(new Uint8Array(4)) }],
  ])('rejects %s', (_label, over) => {
    expect(parseDepth(depthMsg(over), 0)).toBeNull()
  })
})
