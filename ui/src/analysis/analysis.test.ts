import { describe, expect, it, vi } from 'vitest'
import { GH, GW, assumedIntrinsics, type FrameMeta } from '../types'
import { unavailableAnalyzer } from './analyzer'

const meta = (): FrameMeta => ({
  source: 'upload', frameId: 't', stamp: Date.now(), receivedAt: Date.now(),
  width: 640, height: 480, K: assumedIntrinsics(640, 480), kAssumed: true, streaming: false,
})

describe('production analyzer seam', () => {
  it('reports unavailable and never fabricates a result', async () => {
    expect(unavailableAnalyzer.available).toBe(false)
    expect(await unavailableAnalyzer.analyze({} as ImageBitmap, meta())).toBeNull()
  })
})

describe('mock fixture (test-only)', () => {
  it('produces a legal Perception Port result', async () => {
    // the fixture draws through a canvas; give it a flat grey frame instead of a real 2D context
    const data = new Uint8ClampedArray(GW * GH * 4).fill(128)
    vi.stubGlobal('document', {
      createElement: () => ({
        width: 0, height: 0,
        getContext: () => ({ drawImage: () => {}, getImageData: () => ({ data }) }),
      }),
    })
    const { analyze } = await import('./__fixtures__/mock')
    const a = analyze({} as CanvasImageSource, meta())
    expect(a.mock).toBe(true)
    expect(a.mask).toHaveLength(GW * GH)
    expect([...new Set(a.mask)].every((c) => c === 0 || c === 1 || c === 2)).toBe(true)
    vi.unstubAllGlobals()
  })
})

describe('mock isolation', () => {
  it('is imported only by test files', () => {
    const sources = import.meta.glob('/src/**/*.{ts,tsx}', { query: '?raw', import: 'default', eager: true }) as Record<string, string>
    const offenders = Object.entries(sources)
      .filter(([path]) => !/\.test\.tsx?$/.test(path) && !path.includes('/__fixtures__/'))
      .filter(([, code]) => /analysis\/mock|__fixtures__/.test(code))
      .map(([path]) => path)
    expect(offenders).toEqual([])
  })
})
