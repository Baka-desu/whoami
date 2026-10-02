import { describe, expect, it } from 'vitest'
import type { DepthFrame, ElevationFrame, GridFrame } from './codec'
import {
  buildElevationGeometry, buildGridTexture, colorByHeight, colorElevation, depthToRgba, writeRamp,
} from './geometry'

const NaN_ = Number.NaN

// An elevation frame from row-major heights (row 0 first). Obstacle and confidence default to 0 / 128.
function elevation(width: number, height: number, heightM: number[], over: Partial<ElevationFrame> = {}): ElevationFrame {
  const n = width * height
  return {
    epoch: 1, seq: 1, stampS: 0, width, height, resolution: 0.5, originX: 0, originY: 0,
    knownCells: heightM.filter((h) => Number.isFinite(h)).length,
    heightM: Float32Array.from(heightM),
    obstacle: new Uint8Array(n),
    confidence: new Uint8Array(n).fill(128),
    ...over,
  }
}

const flat = (width: number, height: number, z = 1) => elevation(width, height, new Array<number>(width * height).fill(z))

function rampAt(t: number): number[] {
  const out = new Uint8Array(3)
  writeRamp(t, out, 0)
  return Array.from(out)
}

const rgbAt = (colors: Uint8Array | Uint8ClampedArray, i: number, stride = 3) => Array.from(colors.subarray(i * stride, i * stride + 3))
const luminance = ([r, g, b]: number[]) => 0.2126 * r + 0.7152 * g + 0.0722 * b

describe('colour ramp', () => {
  it('ends are distinct and the middle sits between them', () => {
    const lo = rampAt(0)
    const hi = rampAt(1)
    expect(lo).not.toEqual(hi)
    expect(rampAt(0.5)).not.toEqual(lo)
    expect(rampAt(0.5)).not.toEqual(hi)
  })

  it('is perceptually ordered: luminance never decreases from low to high', () => {
    let prev = -1
    for (let i = 0; i <= 100; i++) {
      const y = luminance(rampAt(i / 100))
      expect(y).toBeGreaterThanOrEqual(prev - 0.5) // rounding to bytes may wobble by well under 1
      prev = y
    }
    expect(luminance(rampAt(1))).toBeGreaterThan(luminance(rampAt(0)) + 100)
  })

  it('clamps outside 0..1 and sends non-finite input to the low colour', () => {
    expect(rampAt(-5)).toEqual(rampAt(0))
    expect(rampAt(7)).toEqual(rampAt(1))
    expect(rampAt(NaN_)).toEqual(rampAt(0))
    expect(rampAt(Infinity)).toEqual(rampAt(1))
  })

  it('writes three bytes at the offset and nothing else', () => {
    const out = new Uint8Array(9).fill(9)
    writeRamp(1, out, 3)
    expect(Array.from(out.subarray(0, 3))).toEqual([9, 9, 9])
    expect(Array.from(out.subarray(3, 6))).toEqual(rampAt(1))
    expect(Array.from(out.subarray(6))).toEqual([9, 9, 9])
  })
})

describe('buildElevationGeometry: vertices', () => {
  it('puts one vertex at the centre of every known cell: x along columns, y along rows, z up', () => {
    // 3 columns x 2 rows, origin (10, -5), 0.5 m cells; heights are the flat index so each vertex is identifiable
    const f = elevation(3, 2, [0, 1, 2, 3, 4, 5], { originX: 10, originY: -5, resolution: 0.5 })
    const g = buildElevationGeometry(f)
    expect(g.positions).toBeInstanceOf(Float32Array)
    expect(g.positions.length).toBe(18)
    expect(Array.from(g.cellOfVertex)).toEqual([0, 1, 2, 3, 4, 5])
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 3; c++) {
        const v = r * 3 + c
        expect(g.positions[3 * v]).toBeCloseTo(10 + (c + 0.5) * 0.5, 5)
        expect(g.positions[3 * v + 1]).toBeCloseTo(-5 + (r + 0.5) * 0.5, 5)
        expect(g.positions[3 * v + 2]).toBeCloseTo(v, 5)
      }
    }
  })

  it('skips unknown cells but keeps cellOfVertex pointing at the flat cell index', () => {
    const f = elevation(3, 3, [0, 1, 2, 3, NaN_, 5, 6, 7, 8])
    const g = buildElevationGeometry(f)
    expect(g.positions.length).toBe(8 * 3)
    expect(Array.from(g.cellOfVertex)).toEqual([0, 1, 2, 3, 5, 6, 7, 8])
    expect(g.cellOfVertex).toBeInstanceOf(Uint32Array)
    // the vertex of cell 5 (row 1, col 2) is vertex 4
    expect(g.positions[3 * 4]).toBeCloseTo((2 + 0.5) * 0.5, 5)
    expect(g.positions[3 * 4 + 1]).toBeCloseTo((1 + 0.5) * 0.5, 5)
    expect(g.positions[3 * 4 + 2]).toBeCloseTo(5, 5)
  })

  it('treats infinite heights as unknown too', () => {
    const f = elevation(2, 1, [Infinity, 1])
    expect(Array.from(buildElevationGeometry(f).cellOfVertex)).toEqual([1])
  })

  it('raises obstacle cells by obstacle * 0.05 m and leaves the others at their terrain height', () => {
    const f = elevation(2, 2, [1, 1, 1, 1], { obstacle: Uint8Array.from([0, 4, 0, 255]) })
    const { positions } = buildElevationGeometry(f)
    expect(positions[2]).toBeCloseTo(1, 5)
    expect(positions[5]).toBeCloseTo(1 + 4 * 0.05, 5)
    expect(positions[8]).toBeCloseTo(1, 5)
    expect(positions[11]).toBeCloseTo(1 + 255 * 0.05, 4)
  })

  it('returns empty typed arrays, not null, for an empty frame', () => {
    for (const f of [elevation(0, 0, []), elevation(0, 3, []), elevation(3, 0, []), elevation(2, 2, [NaN_, NaN_, NaN_, NaN_])]) {
      const g = buildElevationGeometry(f)
      expect(g.positions).toBeInstanceOf(Float32Array)
      expect(g.indices).toBeInstanceOf(Uint32Array)
      expect(g.cellOfVertex).toBeInstanceOf(Uint32Array)
      expect(g.positions.length).toBe(0)
      expect(g.indices.length).toBe(0)
      expect(g.cellOfVertex.length).toBe(0)
    }
  })
})

describe('buildElevationGeometry: triangles', () => {
  it('a fully known 3x3 grid has 4 quads, 8 triangles', () => {
    const g = buildElevationGeometry(flat(3, 3))
    expect(g.indices).toBeInstanceOf(Uint32Array)
    expect(g.indices.length).toBe(4 * 6)
  })

  it('a NaN centre cell leaves a hole: no quad touches it, so no triangles at all', () => {
    const f = elevation(3, 3, [1, 1, 1, 1, NaN_, 1, 1, 1, 1])
    const g = buildElevationGeometry(f)
    expect(g.cellOfVertex.length).toBe(8)
    expect(g.indices.length).toBe(0)
  })

  it('a NaN corner cell removes only the one quad that contains it', () => {
    const f = elevation(3, 3, [NaN_, 1, 1, 1, 1, 1, 1, 1, 1])
    const g = buildElevationGeometry(f)
    expect(g.indices.length).toBe(3 * 6)
    // no triangle may use a cell that is not known: every index is a real vertex
    for (const i of g.indices) expect(i).toBeLessThan(g.cellOfVertex.length)
  })

  it('a hole in a larger grid removes exactly the four blocks around it', () => {
    const h = new Array<number>(16).fill(1)
    h[1 * 4 + 1] = NaN_
    const g = buildElevationGeometry(elevation(4, 4, h))
    expect(g.indices.length).toBe((9 - 4) * 6)
    for (const i of g.indices) {
      expect(i).toBeLessThan(g.cellOfVertex.length)
      expect(g.cellOfVertex[i]).not.toBe(1 * 4 + 1)
    }
  })

  it.each([0, 1, 2, 3])('a 2x2 block with its cell %i unknown has no triangle', (hole) => {
    const h = [1, 1, 1, 1]
    h[hole] = NaN_
    const g = buildElevationGeometry(elevation(2, 2, h))
    expect(g.cellOfVertex.length).toBe(3)
    expect(g.indices.length).toBe(0)
  })

  it.each([0, 1, 3, 4])('an unknown cell %i of a 3x3 grid never appears in a triangle', (cellOfCorner) => {
    const h = new Array<number>(9).fill(1)
    h[cellOfCorner] = NaN_
    const g = buildElevationGeometry(elevation(3, 3, h))
    const used = new Set<number>()
    for (const i of g.indices) used.add(g.cellOfVertex[i])
    expect(used.has(cellOfCorner)).toBe(false)
    for (const i of g.indices) expect(i).toBeLessThan(g.cellOfVertex.length)
    for (let t = 0; t < g.indices.length; t += 3) {
      const cells = [g.indices[t], g.indices[t + 1], g.indices[t + 2]].map((v) => g.cellOfVertex[v])
      expect(cells.every((c) => Number.isFinite(h[c]))).toBe(true)
    }
  })

  it('every triangle is counter-clockwise seen from +z and joins neighbouring cells of one 2x2 block', () => {
    const f = elevation(4, 3, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], { originX: 2, originY: 3, resolution: 0.25 })
    const { positions, indices, cellOfVertex } = buildElevationGeometry(f)
    expect(indices.length).toBe(3 * 2 * 6)
    for (let t = 0; t < indices.length; t += 3) {
      const [a, b, c] = [indices[t], indices[t + 1], indices[t + 2]]
      const ax = positions[3 * a]
      const ay = positions[3 * a + 1]
      const area2 = (positions[3 * b] - ax) * (positions[3 * c + 1] - ay) - (positions[3 * c] - ax) * (positions[3 * b + 1] - ay)
      expect(area2).toBeGreaterThan(0) // CCW seen from +z
      expect(area2 / 2).toBeCloseTo(0.25 * 0.25 / 2, 5) // half a cell
      const cells = [a, b, c].map((v) => cellOfVertex[v])
      const rows = cells.map((x) => Math.floor(x / 4))
      const cols = cells.map((x) => x % 4)
      expect(Math.max(...rows) - Math.min(...rows)).toBe(1)
      expect(Math.max(...cols) - Math.min(...cols)).toBe(1)
    }
  })

  it('covers each quad with exactly two triangles (no overlap, no gap)', () => {
    const g = buildElevationGeometry(flat(5, 4))
    const total = g.indices.length / 3
    expect(total).toBe(4 * 3 * 2)
    let area = 0
    for (let t = 0; t < g.indices.length; t += 3) {
      const [a, b, c] = [g.indices[t], g.indices[t + 1], g.indices[t + 2]]
      const p = g.positions
      area += ((p[3 * b] - p[3 * a]) * (p[3 * c + 1] - p[3 * a + 1]) - (p[3 * c] - p[3 * a]) * (p[3 * b + 1] - p[3 * a + 1])) / 2
    }
    expect(area).toBeCloseTo(4 * 3 * 0.5 * 0.5, 4)
  })

  it('handles the largest elevation frame (512 x 512) with the exact vertex and index counts', () => {
    const g = buildElevationGeometry(flat(512, 512))
    expect(g.positions.length).toBe(512 * 512 * 3)
    expect(g.cellOfVertex.length).toBe(512 * 512)
    expect(g.indices.length).toBe(511 * 511 * 6)
    expect(g.cellOfVertex[512 * 512 - 1]).toBe(512 * 512 - 1)
  })
})

describe('colorElevation', () => {
  const frame = () =>
    elevation(3, 2, [0, 1, 2, 3, 4, 5], {
      obstacle: Uint8Array.from([0, 0, 0, 0, 2, 0]),
      confidence: Uint8Array.from([0, 51, 102, 153, 204, 255]),
    })

  it('returns three bytes per vertex', () => {
    const f = frame()
    const { cellOfVertex } = buildElevationGeometry(f)
    for (const mode of ['height', 'confidence', 'obstacle'] as const) {
      const c = colorElevation(f, cellOfVertex, mode)
      expect(c).toBeInstanceOf(Uint8Array)
      expect(c.length).toBe(cellOfVertex.length * 3)
    }
    expect(colorElevation(f, new Uint32Array(0), 'height').length).toBe(0)
  })

  it('height: the ramp over the finite min..max of the displayed z, obstacle raise included', () => {
    const f = frame() // cell 4: terrain 4 + 2 * 0.05 = 4.1, cell 5 is the max at 5
    const { cellOfVertex } = buildElevationGeometry(f)
    const c = colorElevation(f, cellOfVertex, 'height')
    expect(rgbAt(c, 0)).toEqual(rampAt(0)) // z = 0 is the minimum
    expect(rgbAt(c, 5)).toEqual(rampAt(1)) // z = 5 is the maximum
    expect(rgbAt(c, 4)).toEqual(rampAt(4.1 / 5))
    expect(rgbAt(c, 2)).toEqual(rampAt(2 / 5))
  })

  it('height: an obstacle that stands above the tallest terrain defines the top of the ramp', () => {
    const f = elevation(2, 1, [1, 1], { obstacle: Uint8Array.from([0, 20]) }) // 1.0 and 2.0 displayed
    const c = colorElevation(f, buildElevationGeometry(f).cellOfVertex, 'height')
    expect(rgbAt(c, 0)).toEqual(rampAt(0))
    expect(rgbAt(c, 1)).toEqual(rampAt(1))
  })

  it('height: a flat frame (min == max) gets one constant mid colour', () => {
    const f = flat(2, 2, 3)
    const c = colorElevation(f, buildElevationGeometry(f).cellOfVertex, 'height')
    for (let v = 0; v < 4; v++) expect(rgbAt(c, v)).toEqual(rampAt(0.5))
  })

  it('height: unknown cells do not take part in the range', () => {
    const f = elevation(3, 1, [NaN_, 10, 20])
    const c = colorElevation(f, buildElevationGeometry(f).cellOfVertex, 'height')
    expect(rgbAt(c, 0)).toEqual(rampAt(0)) // cell 1, 10 m
    expect(rgbAt(c, 1)).toEqual(rampAt(1)) // cell 2, 20 m
  })

  it('confidence: dark to bright over 0..255', () => {
    const f = frame()
    const c = colorElevation(f, buildElevationGeometry(f).cellOfVertex, 'confidence')
    expect(rgbAt(c, 0)).toEqual(rampAt(0))
    expect(rgbAt(c, 5)).toEqual(rampAt(1))
    expect(rgbAt(c, 3)).toEqual(rampAt(153 / 255))
    expect(luminance(rgbAt(c, 5))).toBeGreaterThan(luminance(rgbAt(c, 0)))
  })

  it('confidence follows the cell of each vertex when a hole shifts the vertex numbering', () => {
    const f = elevation(3, 1, [NaN_, 1, 1], { confidence: Uint8Array.from([255, 0, 255]) })
    const { cellOfVertex } = buildElevationGeometry(f)
    expect(Array.from(cellOfVertex)).toEqual([1, 2])
    const c = colorElevation(f, cellOfVertex, 'confidence')
    expect(rgbAt(c, 0)).toEqual(rampAt(0))
    expect(rgbAt(c, 1)).toEqual(rampAt(1))
  })

  it('obstacle: obstacle cells share one warning colour, every other cell one neutral ground colour', () => {
    const f = elevation(3, 2, [0, 1, 2, 3, 4, 5], { obstacle: Uint8Array.from([0, 7, 0, 1, 0, 255]) })
    const c = colorElevation(f, buildElevationGeometry(f).cellOfVertex, 'obstacle')
    const warning = rgbAt(c, 1)
    const ground = rgbAt(c, 0)
    expect(warning).not.toEqual(ground)
    expect(rgbAt(c, 3)).toEqual(warning)
    expect(rgbAt(c, 5)).toEqual(warning)
    expect(rgbAt(c, 2)).toEqual(ground)
    expect(rgbAt(c, 4)).toEqual(ground)
    // the warning colour reads as warm (red dominates); the ground colour is a neutral, low-saturation tone
    expect(warning[0]).toBeGreaterThan(warning[2] + 80)
    expect(Math.max(...ground) - Math.min(...ground)).toBeLessThan(40)
  })

  it('switching colour mode changes only the colour array: positions, indices and the frame stay as they were', () => {
    const f = frame()
    const g = buildElevationGeometry(f)
    const positions = g.positions.slice()
    const indices = g.indices.slice()
    const cells = g.cellOfVertex.slice()
    const frameCopy = {
      h: f.heightM.slice(), o: f.obstacle.slice(), c: f.confidence.slice(),
    }
    const height = colorElevation(f, g.cellOfVertex, 'height')
    const confidence = colorElevation(f, g.cellOfVertex, 'confidence')
    const obstacle = colorElevation(f, g.cellOfVertex, 'obstacle')
    expect(Array.from(height)).not.toEqual(Array.from(confidence))
    expect(Array.from(confidence)).not.toEqual(Array.from(obstacle))
    expect(Array.from(height)).not.toEqual(Array.from(obstacle))
    expect(Array.from(g.positions)).toEqual(Array.from(positions))
    expect(Array.from(g.indices)).toEqual(Array.from(indices))
    expect(Array.from(g.cellOfVertex)).toEqual(Array.from(cells))
    expect(Array.from(f.heightM)).toEqual(Array.from(frameCopy.h))
    expect(Array.from(f.obstacle)).toEqual(Array.from(frameCopy.o))
    expect(Array.from(f.confidence)).toEqual(Array.from(frameCopy.c))
    // each call allocates its own result
    expect(height.buffer).not.toBe(confidence.buffer)
  })
})

describe('colorByHeight', () => {
  const pts = (...z: number[]) => Float32Array.from(z.flatMap((v, i) => [i, -i, v]))

  it('colours each point by its z from the shared ramp', () => {
    const c = colorByHeight(pts(0, 1, 2, 4), 0, 4)
    expect(c).toBeInstanceOf(Uint8Array)
    expect(c.length).toBe(12)
    expect(rgbAt(c, 0)).toEqual(rampAt(0))
    expect(rgbAt(c, 1)).toEqual(rampAt(0.25))
    expect(rgbAt(c, 2)).toEqual(rampAt(0.5))
    expect(rgbAt(c, 3)).toEqual(rampAt(1))
  })

  it('clamps z outside zMin..zMax', () => {
    const c = colorByHeight(pts(-10, 10), 0, 1)
    expect(rgbAt(c, 0)).toEqual(rampAt(0))
    expect(rgbAt(c, 1)).toEqual(rampAt(1))
  })

  it('gives every point the mid colour when zMax <= zMin', () => {
    for (const [lo, hi] of [[2, 2], [3, 1]]) {
      const c = colorByHeight(pts(0, 2, 9), lo, hi)
      for (let i = 0; i < 3; i++) expect(rgbAt(c, i)).toEqual(rampAt(0.5))
    }
  })

  it('gives non-finite z the low colour', () => {
    const c = colorByHeight(pts(NaN_, Infinity, -Infinity, 2), 0, 4)
    expect(rgbAt(c, 0)).toEqual(rampAt(0))
    expect(rgbAt(c, 1)).toEqual(rampAt(0))
    expect(rgbAt(c, 2)).toEqual(rampAt(0))
    expect(rgbAt(c, 3)).toEqual(rampAt(0.5))
  })

  it('uses the same ramp as the elevation height mode', () => {
    const f = elevation(2, 1, [0, 2])
    const fromElevation = colorElevation(f, buildElevationGeometry(f).cellOfVertex, 'height')
    const fromCloud = colorByHeight(buildElevationGeometry(f).positions, 0, 2)
    expect(Array.from(fromCloud)).toEqual(Array.from(fromElevation))
  })

  it('returns an empty array for no points and handles 500k points', () => {
    expect(colorByHeight(new Float32Array(0), 0, 1).length).toBe(0)
    const n = 500_000
    const xyz = new Float32Array(3 * n)
    for (let i = 0; i < n; i++) xyz[3 * i + 2] = i / n
    const c = colorByHeight(xyz, 0, 1)
    expect(c.length).toBe(3 * n)
    expect(rgbAt(c, 0)).toEqual(rampAt(0))
    expect(luminance(rgbAt(c, n - 1))).toBeGreaterThan(luminance(rgbAt(c, 0)))
  })
})

function grid(width: number, height: number, cells: number[]): GridFrame {
  return { epoch: 1, seq: 1, stampS: 0, width, height, resolution: 0.05, originX: 0, originY: 0, originYaw: 0, cells: Int8Array.from(cells) }
}

const texel = (t: Uint8ClampedArray, i: number) => Array.from(t.subarray(4 * i, 4 * i + 4))

describe('buildGridTexture', () => {
  it('is RGBA, width * height * 4, as a Uint8ClampedArray', () => {
    const t = buildGridTexture(grid(3, 2, [0, 0, 0, 0, 0, 0]))
    expect(t).toBeInstanceOf(Uint8ClampedArray)
    expect(t.length).toBe(3 * 2 * 4)
  })

  it('makes unknown (-1) and free (0) cells fully transparent', () => {
    const t = buildGridTexture(grid(2, 1, [-1, 0]))
    expect(texel(t, 0)[3]).toBe(0)
    expect(texel(t, 1)[3]).toBe(0)
  })

  it('treats any other negative value as unknown', () => {
    const t = buildGridTexture(grid(2, 1, [-2, -128]))
    expect(texel(t, 0)[3]).toBe(0)
    expect(texel(t, 1)[3]).toBe(0)
  })

  it('draws the inflation halo (1..98) translucent, getting stronger with the cost', () => {
    const t = buildGridTexture(grid(4, 1, [1, 30, 70, 98]))
    const alphas = [0, 1, 2, 3].map((i) => texel(t, i)[3])
    for (const a of alphas) {
      expect(a).toBeGreaterThan(0)
      expect(a).toBeLessThan(255)
    }
    expect(alphas[1]).toBeGreaterThan(alphas[0])
    expect(alphas[2]).toBeGreaterThan(alphas[1])
    expect(alphas[3]).toBeGreaterThan(alphas[2])
  })

  it('draws 99 (inscribed) and 100 (lethal) opaque in two different strong colours', () => {
    const t = buildGridTexture(grid(4, 1, [98, 99, 100, 50]))
    expect(texel(t, 1)[3]).toBe(255)
    expect(texel(t, 2)[3]).toBe(255)
    expect(texel(t, 1).slice(0, 3)).not.toEqual(texel(t, 2).slice(0, 3))
    expect(texel(t, 0)[3]).toBeLessThan(255)
    // lethal reads as the hottest colour (red channel above the others)
    const [r, g, b] = texel(t, 2)
    expect(r).toBeGreaterThan(g)
    expect(r).toBeGreaterThan(b)
  })

  it('shares the ramp with the height colours for the halo', () => {
    const t = buildGridTexture(grid(1, 1, [50]))
    expect(texel(t, 0).slice(0, 3)).toEqual(rampAt(0.5))
  })

  it('treats a value above 100 as lethal rather than dropping it', () => {
    const t = buildGridTexture(grid(1, 1, [127]))
    expect(texel(t, 0)).toEqual(texel(buildGridTexture(grid(1, 1, [100])), 0))
  })

  it('keeps texture row 0 = grid row 0, row-major (a cell at row 1, col 2 of a 3 x 2 grid is texel 5)', () => {
    const cells = [0, 0, 0, 0, 0, 100]
    const t = buildGridTexture(grid(3, 2, cells))
    for (let i = 0; i < 5; i++) expect(texel(t, i)[3]).toBe(0)
    expect(texel(t, 5)[3]).toBe(255)
    const top = buildGridTexture(grid(3, 2, [100, 0, 0, 0, 0, 0]))
    expect(texel(top, 0)[3]).toBe(255)
  })

  it('returns an empty array for an empty grid', () => {
    expect(buildGridTexture(grid(0, 0, [])).length).toBe(0)
  })

  it('handles a 512 x 512 grid', () => {
    const cells = new Array<number>(512 * 512).fill(-1)
    cells[512 * 512 - 1] = 100
    const t = buildGridTexture(grid(512, 512, cells))
    expect(t.length).toBe(512 * 512 * 4)
    expect(t[t.length - 1]).toBe(255)
  })
})

function depth(width: number, height: number, counts: number[], over: Partial<DepthFrame> = {}): DepthFrame {
  return { epoch: 1, seq: 1, stampS: 0, width, height, unitM: 0.001, maxRangeM: 8, counts: Uint16Array.from(counts), ...over }
}

describe('depthToRgba', () => {
  it('is RGBA, width * height * 4, as a Uint8ClampedArray', () => {
    const t = depthToRgba(depth(3, 2, [1, 2, 3, 4, 5, 6]))
    expect(t).toBeInstanceOf(Uint8ClampedArray)
    expect(t.length).toBe(24)
  })

  it('makes holes (count 0) fully transparent', () => {
    const t = depthToRgba(depth(2, 1, [0, 1000]))
    expect(texel(t, 0)).toEqual([0, 0, 0, 0])
    expect(texel(t, 1)[3]).toBe(255)
  })

  it('is a grey ramp over 0..maxRangeM, near bright and far dark', () => {
    // 0.5 m, 2 m, 4 m, 8 m, with 8 m max range and 1 mm counts
    const t = depthToRgba(depth(4, 1, [500, 2000, 4000, 8000]))
    const greys = [0, 1, 2, 3].map((i) => texel(t, i))
    for (const [r, g, b, a] of greys) {
      expect(r).toBe(g)
      expect(g).toBe(b)
      expect(a).toBe(255)
    }
    expect(greys[0][0]).toBeGreaterThan(greys[1][0])
    expect(greys[1][0]).toBeGreaterThan(greys[2][0])
    expect(greys[2][0]).toBeGreaterThan(greys[3][0])
    expect(greys[3][0]).toBe(0) // at max range
    expect(greys[2][0]).toBeCloseTo(128, -1) // half range, about mid grey
  })

  it('converts counts to metres with unitM, not with a fixed 1 mm', () => {
    const a = depthToRgba(depth(1, 1, [100], { unitM: 0.01 })) // 1 m
    const b = depthToRgba(depth(1, 1, [1000], { unitM: 0.001 })) // 1 m
    expect(texel(a, 0)).toEqual(texel(b, 0))
  })

  it('clamps beyond the maximum range to the far colour and keeps the pixel visible', () => {
    const t = depthToRgba(depth(1, 1, [65535], { maxRangeM: 8 })) // 65.5 m
    expect(texel(t, 0)).toEqual([0, 0, 0, 255])
  })

  it('stays row-major: image row 0 first', () => {
    const t = depthToRgba(depth(2, 2, [0, 0, 0, 1000]))
    expect(texel(t, 3)[3]).toBe(255)
    for (let i = 0; i < 3; i++) expect(texel(t, i)[3]).toBe(0)
  })

  it('does not throw on a degenerate range and still marks holes transparent', () => {
    const t = depthToRgba(depth(2, 1, [0, 1000], { maxRangeM: 0 }))
    expect(texel(t, 0)[3]).toBe(0)
    expect(texel(t, 1)[3]).toBe(255)
  })

  it('returns an empty array for an empty frame and handles 640 x 480', () => {
    expect(depthToRgba(depth(0, 0, [])).length).toBe(0)
    const counts = new Uint16Array(640 * 480).fill(2500)
    const t = depthToRgba({ ...depth(640, 480, []), counts })
    expect(t.length).toBe(640 * 480 * 4)
  })
})
