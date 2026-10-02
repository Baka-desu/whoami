// Pure geometry and colour builders for the 3D map view. They turn the decoded layer frames (codec.ts) into typed
// arrays a three.js scene can upload as they are, and they import nothing from three.js, so they run (and are
// tested) without WebGL. Every builder is one or two passes over typed arrays and allocates only its result:
// they run on the main thread for up to 512 x 512 cells and 500 000 points.
//
// Conventions the scene must follow:
//   World frame   map coordinates in metres, x east-ish, y north-ish, z up.
//   Elevation     cell (row r, column c) covers [originX + c*res, originX + (c+1)*res] x [originY + r*res,
//                 originY + (r+1)*res]; its one sample sits at the cell centre. Triangles are counter-clockwise seen
//                 from +z (front faces look up). Unknown cells are holes: no vertex and no triangle touches them.
//   Textures      row 0 of a texture is row 0 of the grid / depth image, row-major, RGBA. For the cost grid that is
//                 the row at originY, so a texture with flipY = false laid on a plane whose v axis follows +y needs
//                 no flip. Alpha is straight (not premultiplied).
//   Colours       one ramp for everything (writeRamp): dark purple (low) through teal to yellow (high). Bytes are
//                 sRGB, three per vertex or point.

import type { DepthFrame, ElevationFrame, GridFrame } from './codec'

// ---- the one colour ramp --------------------------------------------------------------------------
// A perceptually ordered ramp (the viridis samples at eighths), lightness rising monotonically from low to high.
const RAMP_STOPS: readonly (readonly [number, number, number])[] = [
  [68, 1, 84], [71, 44, 122], [59, 82, 139], [44, 113, 142], [33, 144, 141],
  [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37],
]

const RAMP_STEPS = 256
const RAMP_LUT = (() => {
  const lut = new Uint8Array(RAMP_STEPS * 3)
  const last = RAMP_STOPS.length - 1
  for (let i = 0; i < RAMP_STEPS; i++) {
    const pos = (i / (RAMP_STEPS - 1)) * last
    const lo = Math.min(Math.floor(pos), last - 1)
    const k = pos - lo
    for (let ch = 0; ch < 3; ch++) lut[3 * i + ch] = Math.round(RAMP_STOPS[lo][ch] + (RAMP_STOPS[lo + 1][ch] - RAMP_STOPS[lo][ch]) * k)
  }
  return lut
})()

type Bytes = Uint8Array | Uint8ClampedArray

// Writes the ramp colour for t in [0, 1] as three bytes at out[offset..offset + 2]. t is clamped; NaN reads as 0.
export function writeRamp(t: number, out: Bytes, offset: number): void {
  const i = t > 0 ? (t < 1 ? Math.round(t * (RAMP_STEPS - 1)) : RAMP_STEPS - 1) : 0
  out[offset] = RAMP_LUT[3 * i]
  out[offset + 1] = RAMP_LUT[3 * i + 1]
  out[offset + 2] = RAMP_LUT[3 * i + 2]
}

// Height colour of z over zMin..zMax: clamped; the mid colour when the range is empty (zMax <= zMin or not a
// number); the low colour for a z that is not finite.
function writeHeightColor(z: number, zMin: number, zMax: number, out: Bytes, offset: number): void {
  if (!Number.isFinite(z)) writeRamp(0, out, offset)
  else if (!(zMax > zMin)) writeRamp(0.5, out, offset)
  else writeRamp((z - zMin) / (zMax - zMin), out, offset)
}

// ---- elevation mesh -------------------------------------------------------------------------------
export const OBSTACLE_UNIT_M = 0.05 // the wire's obstacle byte counts 5 cm steps

export interface ElevationGeometry {
  positions: Float32Array // x y z per vertex, one vertex per known cell
  indices: Uint32Array // three per triangle, counter-clockwise seen from +z
  cellOfVertex: Uint32Array // flat cell index (row * width + column) of each vertex
}

// One vertex per known cell, at the cell centre, z = height plus 5 cm per obstacle unit so obstacles stand up from
// the terrain. Two triangles for every 2 x 2 block of cells that are all known; a block that touches an unknown
// cell gets none, so unknown regions stay holes.
export function buildElevationGeometry(f: ElevationFrame): ElevationGeometry {
  const { width: w, height: h, heightM, obstacle, resolution: res, originX, originY } = f
  const cells = w * h

  let vertices = 0
  for (let i = 0; i < cells; i++) if (Number.isFinite(heightM[i])) vertices++
  const positions = new Float32Array(3 * vertices)
  const cellOfVertex = new Uint32Array(vertices)
  const vertexOfCell = new Int32Array(cells) // -1 = unknown cell
  let v = 0
  for (let r = 0; r < h; r++) {
    const y = originY + (r + 0.5) * res
    for (let c = 0; c < w; c++) {
      const i = r * w + c
      const z = heightM[i]
      if (!Number.isFinite(z)) {
        vertexOfCell[i] = -1
        continue
      }
      vertexOfCell[i] = v
      cellOfVertex[v] = i
      positions[3 * v] = originX + (c + 0.5) * res
      positions[3 * v + 1] = y
      positions[3 * v + 2] = z + obstacle[i] * OBSTACLE_UNIT_M
      v++
    }
  }

  let quads = 0
  for (let r = 0; r + 1 < h; r++) {
    for (let c = 0; c + 1 < w; c++) {
      const i = r * w + c
      if (vertexOfCell[i] >= 0 && vertexOfCell[i + 1] >= 0 && vertexOfCell[i + w] >= 0 && vertexOfCell[i + w + 1] >= 0) quads++
    }
  }
  const indices = new Uint32Array(6 * quads)
  let k = 0
  for (let r = 0; r + 1 < h; r++) {
    for (let c = 0; c + 1 < w; c++) {
      const i = r * w + c
      const v00 = vertexOfCell[i] // (x, y)
      const v10 = vertexOfCell[i + 1] // (x + res, y)
      const v01 = vertexOfCell[i + w] // (x, y + res)
      const v11 = vertexOfCell[i + w + 1]
      if (v00 < 0 || v10 < 0 || v01 < 0 || v11 < 0) continue
      indices[k++] = v00
      indices[k++] = v10
      indices[k++] = v11
      indices[k++] = v00
      indices[k++] = v11
      indices[k++] = v01
    }
  }
  return { positions, indices, cellOfVertex }
}

export type ElevationColorMode = 'height' | 'confidence' | 'obstacle'

const OBSTACLE_COLOR = [235, 64, 52] as const // warning red-orange
const GROUND_COLOR = [112, 116, 108] as const // neutral grey-green

// RGB bytes, three per vertex, for the mesh that buildElevationGeometry made from `f`. Only colours: it reads the
// frame and never touches positions or indices, so changing the mode is a colour-buffer update.
//   height      the ramp over the finite min..max of the displayed z (obstacle raise included)
//   confidence  the ramp over 0..255, dark = unsure
//   obstacle    obstacle cells one warning colour, every other cell a neutral ground colour
export function colorElevation(f: ElevationFrame, cellOfVertex: Uint32Array, mode: ElevationColorMode): Uint8Array {
  const n = cellOfVertex.length
  const out = new Uint8Array(3 * n)
  if (mode === 'height') {
    const { heightM, obstacle } = f
    let zMin = Infinity
    let zMax = -Infinity
    for (let v = 0; v < n; v++) {
      const cell = cellOfVertex[v]
      const z = heightM[cell] + obstacle[cell] * OBSTACLE_UNIT_M
      if (!Number.isFinite(z)) continue
      if (z < zMin) zMin = z
      if (z > zMax) zMax = z
    }
    for (let v = 0; v < n; v++) {
      const cell = cellOfVertex[v]
      writeHeightColor(heightM[cell] + obstacle[cell] * OBSTACLE_UNIT_M, zMin, zMax, out, 3 * v)
    }
  } else if (mode === 'confidence') {
    const { confidence } = f
    for (let v = 0; v < n; v++) writeRamp(confidence[cellOfVertex[v]] / 255, out, 3 * v)
  } else {
    const { obstacle } = f
    for (let v = 0; v < n; v++) {
      const color = obstacle[cellOfVertex[v]] > 0 ? OBSTACLE_COLOR : GROUND_COLOR
      out[3 * v] = color[0]
      out[3 * v + 1] = color[1]
      out[3 * v + 2] = color[2]
    }
  }
  return out
}

// ---- point clouds ---------------------------------------------------------------------------------
// RGB bytes, three per point, by each point's z (xyz is x y z per point) with the shared ramp over zMin..zMax.
// z is clamped to the range; every point gets the mid colour when zMax <= zMin; a z that is not finite gets the low
// colour.
export function colorByHeight(xyz: Float32Array, zMin: number, zMax: number): Uint8Array {
  const n = Math.floor(xyz.length / 3)
  const out = new Uint8Array(3 * n)
  for (let i = 0; i < n; i++) writeHeightColor(xyz[3 * i + 2], zMin, zMax, out, 3 * i)
  return out
}

// ---- textures -------------------------------------------------------------------------------------
const INSCRIBED_COLOR = [255, 140, 0] as const // cost 99: the footprint would touch an obstacle
const LETHAL_COLOR = [230, 30, 40] as const // cost 100 and above
const HALO_ALPHA_MIN = 40 // the faintest halo cell (cost 1)
const HALO_ALPHA_SPAN = 120 // added up to cost 98

// RGBA texture of a cost grid, width * height * 4, row 0 = grid row 0. Unknown (negative) and free (0) cells are
// fully transparent; costs 1..98, the inflation halo, are translucent and stronger with the cost; 99 (inscribed)
// and 100 (lethal) are opaque strong colours. A value above 100 is not valid and is drawn as lethal rather than
// dropped.
export function buildGridTexture(f: GridFrame): Uint8ClampedArray {
  const n = f.width * f.height
  const { cells } = f
  const out = new Uint8ClampedArray(4 * n)
  for (let i = 0; i < n; i++) {
    const v = cells[i]
    if (v <= 0) continue
    const o = 4 * i
    if (v >= 100) {
      out[o] = LETHAL_COLOR[0]
      out[o + 1] = LETHAL_COLOR[1]
      out[o + 2] = LETHAL_COLOR[2]
      out[o + 3] = 255
    } else if (v === 99) {
      out[o] = INSCRIBED_COLOR[0]
      out[o + 1] = INSCRIBED_COLOR[1]
      out[o + 2] = INSCRIBED_COLOR[2]
      out[o + 3] = 255
    } else {
      writeRamp(v / 100, out, o)
      out[o + 3] = HALO_ALPHA_MIN + (HALO_ALPHA_SPAN * v) / 98
    }
  }
  return out
}

// RGBA grey image of a depth frame, width * height * 4, row 0 first. Brightness falls linearly from white at 0 m to
// black at maxRangeM and beyond (near = bright). Holes (count 0) are fully transparent; if the frame's range is
// not usable every valid pixel is mid grey.
export function depthToRgba(f: DepthFrame): Uint8ClampedArray {
  const n = f.width * f.height
  const { counts } = f
  const out = new Uint8ClampedArray(4 * n)
  const perCount = f.unitM / f.maxRangeM // fraction of the range one count spans
  const usable = Number.isFinite(perCount) && perCount > 0
  for (let i = 0; i < n; i++) {
    const count = counts[i]
    if (count === 0) continue
    const grey = usable ? 255 * (1 - Math.min(1, count * perCount)) : 128
    const o = 4 * i
    out[o] = grey
    out[o + 1] = grey
    out[o + 2] = grey
    out[o + 3] = 255
  }
  return out
}
