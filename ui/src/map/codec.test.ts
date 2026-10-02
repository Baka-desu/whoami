/// <reference types="node" />
// Golden-file tests for the binary map layer decoders. The .bin files are produced by the gateway's own
// encoder (ugv_nav/ugv_api/ugv_api/mapcodec.py) with epoch=7, seq=3, stamp_s=1234.5; their inputs are listed in
// ugv_nav/ugv_api/test/test_mapcodec.py (build_golden). They are read in place, never copied into ui/.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { decodeCloud, decodeGrid, decodeTrajectory } from './codec'

const FIXTURES = new URL('../../../ugv_nav/ugv_api/test/fixtures/map/', import.meta.url)

// An ArrayBuffer holding exactly the file's bytes (a Node Buffer may sit inside a larger pooled ArrayBuffer).
function golden(name: string): ArrayBuffer {
  return Uint8Array.from(readFileSync(new URL(`${name}.bin`, FIXTURES))).buffer
}

const LAYERS = ['cloud', 'trajectory', 'grid'] as const
type Layer = (typeof LAYERS)[number]

const DECODERS: Record<Layer, (buf: ArrayBuffer) => unknown> = {
  cloud: decodeCloud,
  trajectory: decodeTrajectory,
  grid: decodeGrid,
}

// Exact sizes from the spec: prelude 24 + layer header, then the body.
const SIZES: Record<Layer, { total: number; header: number }> = {
  cloud: { total: 184, header: 64 },
  trajectory: { total: 116, header: 32 },
  grid: { total: 60, header: 48 },
}

// A copy of `buf` (optionally cut to `length` bytes) with `edit` applied through a little-endian DataView.
function edited(buf: ArrayBuffer, edit: (dv: DataView) => void = () => {}, length = buf.byteLength): ArrayBuffer {
  const out = buf.slice(0, length)
  edit(new DataView(out))
  return out
}

const prelude = { epoch: 7, seq: 3, stampS: 1234.5 }

describe('golden files', () => {
  it('have the sizes the spec implies', () => {
    for (const layer of LAYERS) expect(golden(layer).byteLength).toBe(SIZES[layer].total)
  })

  it('decodes cloud.bin', () => {
    const buf = golden('cloud')
    const f = decodeCloud(buf)!
    expect(f).not.toBeNull()
    expect(f).toMatchObject({ ...prelude, count: 8, sourceCount: 8, spacingM: 0.25, bboxMin: [-1.5, -0.75, 0], bboxMax: [1, 2.25, 3] })
    expect(Array.from(f.xyz)).toEqual([
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0.5, 1, 0, 0.5, -1.5, 2.25, 3, 0.25, -0.75, 1.5,
    ])
    expect(Array.from(f.rgb!)).toEqual([
      255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 255, 16, 32, 48, 250, 128, 7,
    ])
    // Zero-copy: both arrays are views onto the input at the spec's body offsets.
    expect(f.xyz.buffer).toBe(buf)
    expect(f.xyz.byteOffset).toBe(64)
    expect(f.rgb!.buffer).toBe(buf)
    expect(f.rgb!.byteOffset).toBe(64 + 12 * 8)
  })

  it('decodes trajectory.bin', () => {
    const buf = golden('trajectory')
    const f = decodeTrajectory(buf)!
    expect(f).not.toBeNull()
    expect(f).toMatchObject({ ...prelude, count: 3, lengthM: 17 })
    expect(Array.from(f.poses)).toEqual([0, 0, 0, 0, 0, 0, 1, 3, 4, 0, 0, 0, 1, 0, 3, 4, 12, 0.5, 0.5, 0.5, 0.5])
    expect(f.poses.buffer).toBe(buf)
    expect(f.poses.byteOffset).toBe(32)
  })

  it('decodes grid.bin', () => {
    const buf = golden('grid')
    const f = decodeGrid(buf)!
    expect(f).not.toBeNull()
    expect(f).toMatchObject({ ...prelude, width: 4, height: 3, resolution: 0.25, originX: -0.5, originY: 1, originYaw: 0.5 })
    expect(Array.from(f.cells)).toEqual([-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    expect(f.cells).toBeInstanceOf(Int8Array)
    expect(f.cells.buffer).toBe(buf)
    expect(f.cells.byteOffset).toBe(48)
  })
})

describe.each(LAYERS)('%s decoder rejects bad input', (layer) => {
  const decode = DECODERS[layer]
  const { total, header } = SIZES[layer]

  it('returns null for an empty buffer and for anything shorter than the prelude', () => {
    expect(decode(new ArrayBuffer(0))).toBeNull()
    expect(decode(golden(layer).slice(0, 23))).toBeNull()
  })

  it('returns null for a prelude with no layer header behind it', () => {
    expect(decode(golden(layer).slice(0, 24))).toBeNull()
    expect(decode(golden(layer).slice(0, header - 1))).toBeNull()
  })

  it('returns null for the right magic with the wrong first byte', () => {
    expect(decode(edited(golden(layer), (dv) => dv.setUint8(0, 0x58)))).toBeNull()
  })

  it('returns null for every other layer\'s magic', () => {
    for (const other of LAYERS) if (other !== layer) expect(decode(golden(other))).toBeNull()
  })

  it.each([0, 2, 65535])('returns null for format %i', (format) => {
    expect(decode(edited(golden(layer), (dv) => dv.setUint16(4, format, true)))).toBeNull()
  })

  it('returns null for a header_bytes that differs from the spec', () => {
    for (const bytes of [0, 24, header - 8, header - 4, header + 4, header + 8, 65535]) {
      expect(decode(edited(golden(layer), (dv) => dv.setUint16(6, bytes, true)))).toBeNull()
    }
  })

  it('returns null for a body one byte short', () => {
    expect(decode(golden(layer).slice(0, total - 1))).toBeNull()
  })

  it('returns null for a body one byte long', () => {
    const long = new Uint8Array(total + 1)
    long.set(new Uint8Array(golden(layer)))
    expect(decode(long.buffer)).toBeNull()
  })

  it('accepts the unmodified golden file', () => {
    expect(decode(golden(layer))).not.toBeNull()
  })
})

describe('cloud edge cases', () => {
  it('has no colour when flags bit0 is clear: length 64 + 12 * count', () => {
    const f = decodeCloud(edited(golden('cloud'), (dv) => dv.setUint32(36, 0, true), 64 + 12 * 8))!
    expect(f.count).toBe(8)
    expect(f.rgb).toBeNull()
    expect(f.xyz.length).toBe(24)
    expect(f.xyz[3]).toBe(1)
  })

  it('rejects the rgb length when bit0 is clear, and the xyz-only length when bit0 is set', () => {
    expect(decodeCloud(edited(golden('cloud'), (dv) => dv.setUint32(36, 0, true)))).toBeNull()
    expect(decodeCloud(edited(golden('cloud'), () => {}, 64 + 12 * 8))).toBeNull()
  })

  it('ignores flag bits other than bit0', () => {
    const f = decodeCloud(edited(golden('cloud'), (dv) => dv.setUint32(36, 0b11, true)))!
    expect(f.rgb).not.toBeNull()
    expect(f.rgb!.length).toBe(24)
  })

  it('accepts an empty cloud with the exact header-only length', () => {
    const empty = (flags: number) => edited(golden('cloud'), (dv) => { dv.setUint32(24, 0, true); dv.setUint32(36, flags, true) }, 64)
    const plain = decodeCloud(empty(0))!
    expect(plain.count).toBe(0)
    expect(plain.xyz).toHaveLength(0)
    expect(plain.rgb).toBeNull()
    const coloured = decodeCloud(empty(1))!
    expect(coloured.count).toBe(0)
    expect(coloured.xyz).toHaveLength(0)
    expect(coloured.rgb).toHaveLength(0)
  })

  it('rejects an empty cloud that still carries the old body', () => {
    expect(decodeCloud(edited(golden('cloud'), (dv) => dv.setUint32(24, 0, true)))).toBeNull()
  })

  it('rejects a count that is larger than the body', () => {
    expect(decodeCloud(edited(golden('cloud'), (dv) => dv.setUint32(24, 9, true)))).toBeNull()
    expect(decodeCloud(edited(golden('cloud'), (dv) => dv.setUint32(24, 0xffffffff, true)))).toBeNull()
  })
})

describe('trajectory edge cases', () => {
  it('accepts an empty trajectory with the exact header-only length', () => {
    const f = decodeTrajectory(edited(golden('trajectory'), (dv) => { dv.setUint32(24, 0, true); dv.setFloat32(28, 0, true) }, 32))!
    expect(f).toMatchObject({ count: 0, lengthM: 0 })
    expect(f.poses).toHaveLength(0)
  })

  it('rejects a count that disagrees with the body', () => {
    expect(decodeTrajectory(edited(golden('trajectory'), (dv) => dv.setUint32(24, 4, true)))).toBeNull()
  })
})

describe('grid edge cases', () => {
  it('accepts an empty grid with the exact header-only length', () => {
    const f = decodeGrid(edited(golden('grid'), (dv) => { dv.setUint32(24, 0, true); dv.setUint32(28, 0, true) }, 48))!
    expect(f).toMatchObject({ width: 0, height: 0, resolution: 0.25 })
    expect(f.cells).toHaveLength(0)
  })

  it('keeps the signed range: -1 stays -1, 100 stays 100', () => {
    const f = decodeGrid(golden('grid'))!
    expect(f.cells[0]).toBe(-1)
    expect(f.cells[11]).toBe(100)
  })

  it('rejects dimensions that disagree with the body', () => {
    expect(decodeGrid(edited(golden('grid'), (dv) => dv.setUint32(24, 5, true)))).toBeNull()
  })
})

describe('prelude fields', () => {
  it('reads epoch, seq and stamp from their own offsets', () => {
    const buf = edited(golden('grid'), (dv) => {
      dv.setUint32(8, 0xfffffffe, true)
      dv.setUint32(12, 0x01020304, true)
      dv.setFloat64(16, -0.125, true)
    })
    const f = decodeGrid(buf)!
    expect(f.epoch).toBe(0xfffffffe)
    expect(f.seq).toBe(0x01020304)
    expect(f.stampS).toBe(-0.125)
  })
})
