// Decoders for the gateway's binary map layers (cloud, trajectory, cost grid, depth).
// Wire format v1: little-endian, a 24-byte prelude shared by every layer, then a fixed layer header, then the body.
// Every decoder returns null, never throws, for anything that is not exactly a v1 message of its own layer: a buffer
// shorter than the prelude, the wrong magic, a format other than 1, a header_bytes that differs from the spec's value
// for that layer, or a total length other than the one the header implies. A layer with nothing in it (zero points,
// zero cells) is valid when it is exactly header-sized, and decodes to empty arrays.
//
// The bulk arrays are typed-array views onto the input buffer, not copies, so the caller must not mutate or reuse
// the buffer while it still holds the frame. Alignment never forces a copy: a view onto an ArrayBuffer starts at
// byte 0, every Float32Array/Uint16Array body starts at its layer's header size (64, 32, 40, all multiples of 4),
// and the array that follows a float block (cloud rgb) is one byte per element, which has no alignment requirement. The host is little-endian in every supported browser, so the views read the wire
// values directly; the header fields go through a DataView with littleEndian = true.

export type CloudFrame = {
  epoch: number
  seq: number
  stampS: number
  count: number
  sourceCount: number // finite points the gateway had before it cut the cloud to its budget
  spacingM: number // voxel size the gateway used when it cut the cloud
  xyz: Float32Array // 3 * count: x, y, z in metres
  rgb: Uint8Array | null // 3 * count, or null when the cloud carries no colour
  bboxMin: [number, number, number]
  bboxMax: [number, number, number]
}

export type TrajectoryFrame = {
  epoch: number
  seq: number
  stampS: number
  count: number
  lengthM: number
  poses: Float32Array // 7 * count: x, y, z, qx, qy, qz, qw
}

export type GridFrame = {
  epoch: number
  seq: number
  stampS: number
  width: number
  height: number
  resolution: number
  originX: number
  originY: number
  originYaw: number
  cells: Int8Array // width * height, row-major, -1 = unknown, 0..100
}

export type DepthFrame = {
  epoch: number
  seq: number
  stampS: number
  width: number
  height: number
  unitM: number // metres per count
  maxRangeM: number
  counts: Uint16Array // width * height, row-major, 0 = hole
}

const FORMAT = 1
const PRELUDE_BYTES = 24

// Per layer: the four magic bytes and the spec's header_bytes (prelude included). The field offsets used below are
// the spec's, counted from the start of the buffer.
const CLOUD = { magic: 'UGVC', header: 64 }
const TRAJECTORY = { magic: 'UGVT', header: 32 }
const GRID = { magic: 'UGVG', header: 48 }
const DEPTH = { magic: 'UGVD', header: 40 }

type Layer = { magic: string; header: number }

type Opened = { dv: DataView; epoch: number; seq: number; stampS: number }

// Checks everything the layers share and returns a reader for the header fields, or null. A buffer long enough to
// hold the whole layer header is guaranteed when this returns non-null, so the callers' DataView reads cannot throw.
function open(buf: ArrayBuffer, layer: Layer): Opened | null {
  if (buf.byteLength < PRELUDE_BYTES) return null
  const dv = new DataView(buf)
  for (let i = 0; i < 4; i++) if (dv.getUint8(i) !== layer.magic.charCodeAt(i)) return null
  if (dv.getUint16(4, true) !== FORMAT) return null
  if (dv.getUint16(6, true) !== layer.header) return null
  if (buf.byteLength < layer.header) return null
  return { dv, epoch: dv.getUint32(8, true), seq: dv.getUint32(12, true), stampS: dv.getFloat64(16, true) }
}

export function decodeCloud(buf: ArrayBuffer): CloudFrame | null {
  const p = open(buf, CLOUD)
  if (!p) return null
  const { dv } = p
  const count = dv.getUint32(24, true)
  const hasRgb = (dv.getUint32(36, true) & 1) !== 0
  if (buf.byteLength !== CLOUD.header + (hasRgb ? 15 : 12) * count) return null
  return {
    epoch: p.epoch,
    seq: p.seq,
    stampS: p.stampS,
    count,
    sourceCount: dv.getUint32(28, true),
    spacingM: dv.getFloat32(32, true),
    xyz: new Float32Array(buf, CLOUD.header, 3 * count),
    rgb: hasRgb ? new Uint8Array(buf, CLOUD.header + 12 * count, 3 * count) : null,
    bboxMin: [dv.getFloat32(40, true), dv.getFloat32(44, true), dv.getFloat32(48, true)],
    bboxMax: [dv.getFloat32(52, true), dv.getFloat32(56, true), dv.getFloat32(60, true)],
  }
}

export function decodeTrajectory(buf: ArrayBuffer): TrajectoryFrame | null {
  const p = open(buf, TRAJECTORY)
  if (!p) return null
  const { dv } = p
  const count = dv.getUint32(24, true)
  if (buf.byteLength !== TRAJECTORY.header + 28 * count) return null
  return {
    epoch: p.epoch,
    seq: p.seq,
    stampS: p.stampS,
    count,
    lengthM: dv.getFloat32(28, true),
    poses: new Float32Array(buf, TRAJECTORY.header, 7 * count),
  }
}

export function decodeGrid(buf: ArrayBuffer): GridFrame | null {
  const p = open(buf, GRID)
  if (!p) return null
  const { dv } = p
  const width = dv.getUint32(24, true)
  const height = dv.getUint32(28, true)
  const cells = width * height
  if (buf.byteLength !== GRID.header + cells) return null
  return {
    epoch: p.epoch,
    seq: p.seq,
    stampS: p.stampS,
    width,
    height,
    resolution: dv.getFloat32(32, true),
    originX: dv.getFloat32(36, true),
    originY: dv.getFloat32(40, true),
    originYaw: dv.getFloat32(44, true),
    cells: new Int8Array(buf, GRID.header, cells),
  }
}

export function decodeDepth(buf: ArrayBuffer): DepthFrame | null {
  const p = open(buf, DEPTH)
  if (!p) return null
  const { dv } = p
  const width = dv.getUint32(24, true)
  const height = dv.getUint32(28, true)
  const pixels = width * height
  if (buf.byteLength !== DEPTH.header + 2 * pixels) return null
  return {
    epoch: p.epoch,
    seq: p.seq,
    stampS: p.stampS,
    width,
    height,
    unitM: dv.getFloat32(32, true),
    maxRangeM: dv.getFloat32(36, true),
    counts: new Uint16Array(buf, DEPTH.header, pixels),
  }
}
