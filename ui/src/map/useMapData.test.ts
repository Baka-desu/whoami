import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, LAYERS, type Layer, type MapStatus } from '../source/api'
import {
  MIN_INTERVAL_MS, STATUS_POLL_MS, STATUS_STALE_MS, createMapSession, isStale, layerKey, nextFetches, schedule,
  type Frame,
} from './useMapData'

const none = <T>(v: T): Record<Layer, T> => Object.fromEntries(LAYERS.map((l) => [l, v])) as Record<Layer, T>
const only = (...on: Layer[]): Record<Layer, boolean> => ({ ...none(false), ...Object.fromEntries(on.map((l) => [l, true])) })
const ALL = none(true)

const mapStatus = (epoch: number, seq: Partial<Record<Layer, number>> = {}): MapStatus => ({ epoch, seq: { ...none(0), ...seq }, stats: {} })

describe('layerKey', () => {
  it('is the epoch and the layer sequence', () => {
    expect(layerKey(mapStatus(7, { cloud: 3 }), 'cloud')).toBe('7:3')
    expect(layerKey(mapStatus(7, { cloud: 3 }), 'grid')).toBe('7:0')
  })
})

describe('nextFetches', () => {
  it('fetches a layer whose epoch:seq key differs from the one held, and only that one', () => {
    const status = mapStatus(5, { cloud: 3, elevation: 2 })
    const have = { ...none<string | null>(null), cloud: '5:3', elevation: '5:1' }
    expect(nextFetches(status, have, ALL)).toEqual(['elevation'])
  })

  it('fetches nothing when every held key matches', () => {
    const status = mapStatus(5, { cloud: 3, depth: 9 })
    const have = { ...none<string | null>(null), cloud: '5:3', depth: '5:9' }
    expect(nextFetches(status, have, ALL)).toEqual([])
  })

  it('fetches a layer that has never been held', () => {
    expect(nextFetches(mapStatus(5, { grid: 1 }), none<string | null>(null), ALL)).toEqual(['grid'])
  })

  it('refetches every enabled layer when the epoch changes even though no seq did', () => {
    const before = mapStatus(5, { cloud: 3, elevation: 2, trajectory: 4, grid: 1, live: 8, depth: 6, camera: 9 })
    const have = Object.fromEntries(LAYERS.map((l) => [l, layerKey(before, l)])) as Record<Layer, string | null>
    expect(nextFetches(before, have, ALL)).toEqual([])
    const restarted = { ...before, epoch: 6 }
    expect(nextFetches(restarted, have, ALL)).toEqual([...LAYERS])
    expect(nextFetches(restarted, have, only('elevation', 'camera'))).toEqual(['elevation', 'camera'])
  })

  it('never fetches a layer whose seq is 0, whatever is held', () => {
    const status = mapStatus(5, { cloud: 0, grid: 2 })
    expect(nextFetches(status, none<string | null>(null), ALL)).toEqual(['grid'])
    expect(nextFetches(status, { ...none<string | null>(null), cloud: '4:9' }, ALL)).toEqual(['grid'])
  })

  it('never fetches a disabled layer', () => {
    const status = mapStatus(5, { cloud: 1, elevation: 1, camera: 1 })
    expect(nextFetches(status, none<string | null>(null), only('elevation'))).toEqual(['elevation'])
    expect(nextFetches(status, none<string | null>(null), none(false))).toEqual([])
  })

  it('returns layers in LAYERS order', () => {
    const status = mapStatus(1, { camera: 1, depth: 1, cloud: 1, live: 1 })
    expect(nextFetches(status, none<string | null>(null), ALL)).toEqual(['cloud', 'live', 'depth', 'camera'])
  })

  it('does not treat a held key from another epoch with the same seq as current', () => {
    const status = mapStatus(2, { cloud: 3 })
    expect(nextFetches(status, { ...none<string | null>(null), cloud: '1:3' }, ALL)).toEqual(['cloud'])
  })
})

describe('MIN_INTERVAL_MS', () => {
  it('is the specified minimum between fetch starts of one layer', () => {
    expect(MIN_INTERVAL_MS).toEqual({ cloud: 2000, elevation: 1000, trajectory: 1000, grid: 1000, live: 500, depth: 500, camera: 500 })
  })
})

describe('schedule', () => {
  const never = none<number | null>(null)
  const idle = none(false)

  it('starts every wanted layer that has never been started', () => {
    expect(schedule(['cloud', 'depth'], 10_000, never, idle)).toEqual({ start: ['cloud', 'depth'], retryInMs: null })
  })

  it('does nothing when nothing is wanted', () => {
    expect(schedule([], 10_000, never, idle)).toEqual({ start: [], retryInMs: null })
  })

  it('never starts a second fetch for a layer that is in flight, and does not ask to be woken for it', () => {
    const plan = schedule(['cloud', 'grid'], 10_000, never, { ...idle, cloud: true })
    expect(plan).toEqual({ start: ['grid'], retryInMs: null })
  })

  it('holds a layer back until its own minimum interval since the last start has passed', () => {
    const lastStart = { ...never, cloud: 10_000, elevation: 10_000 }
    // 999 ms later elevation (1000) is still too early, cloud (2000) needs 1001 ms more
    expect(schedule(['cloud', 'elevation'], 10_999, lastStart, idle)).toEqual({ start: [], retryInMs: 1 })
    // exactly the interval later elevation may start; cloud still waits another 1000 ms
    expect(schedule(['cloud', 'elevation'], 11_000, lastStart, idle)).toEqual({ start: ['elevation'], retryInMs: 1000 })
    expect(schedule(['cloud'], 12_000, lastStart, idle)).toEqual({ start: ['cloud'], retryInMs: null })
  })

  it('asks to be woken at the earliest moment any held-back layer becomes due', () => {
    const lastStart = { ...never, cloud: 10_000, live: 10_300, camera: 10_100 }
    const plan = schedule(['cloud', 'live', 'camera'], 10_400, lastStart, idle)
    expect(plan.start).toEqual([])
    expect(plan.retryInMs).toBe(200) // live: 10_300 + 500 - 10_400 = 400, camera: 10_100 + 500 - 10_400 = 200
  })

  it('does not count an in-flight layer toward the wake-up even when its interval has not passed', () => {
    const lastStart = { ...never, cloud: 10_000, grid: 10_000 }
    const plan = schedule(['cloud', 'grid'], 10_100, lastStart, { ...idle, grid: true })
    expect(plan).toEqual({ start: [], retryInMs: 1900 })
  })

  it('keeps the order it was given and mixes started and held-back layers', () => {
    const lastStart = { ...never, elevation: 9_900 }
    const plan = schedule(['cloud', 'elevation', 'camera'], 10_000, lastStart, idle)
    expect(plan).toEqual({ start: ['cloud', 'camera'], retryInMs: 900 })
  })

  it('never waits longer than the layer interval if the clock went backwards', () => {
    const plan = schedule(['elevation'], 5_000, { ...never, elevation: 10_000 }, idle)
    expect(plan).toEqual({ start: [], retryInMs: 1000 })
  })
})

describe('isStale', () => {
  it('is true before any poll succeeded', () => {
    expect(isStale(null, false, 10_000)).toBe(true)
  })

  it('is true when the last poll failed, however recent the last success', () => {
    expect(isStale(9_990, true, 10_000)).toBe(true)
  })

  it('is false until the last success is older than 3000 ms', () => {
    expect(STATUS_STALE_MS).toBe(3000)
    expect(isStale(10_000, false, 10_000)).toBe(false)
    expect(isStale(10_000, false, 13_000)).toBe(false)
    expect(isStale(10_000, false, 13_001)).toBe(true)
  })
})

// ---- the session: scheduling, state and cleanup, with fake timers and a fake gateway ----------------
interface Pending {
  path: string
  signal: AbortSignal
  settled: boolean
  resolve: (v: ArrayBuffer | null) => void
  reject: (e: unknown) => void
}

const bytes = (text: string): ArrayBuffer => Uint8Array.from(new TextEncoder().encode(text)).buffer
const statusBody = (epoch: number, seq: Partial<Record<Layer, number>> = {}, stats: Record<string, unknown> = {}) =>
  JSON.stringify({ epoch, seq: { ...none(0), ...seq }, stats })

function harness(initialEnabled: Record<Layer, boolean> = ALL) {
  const pending: Pending[] = []
  const frames: { layer: Layer; text: string }[] = []
  const discarded: { layer: Layer; text: string }[] = []
  const statuses: MapStatus[] = []
  const stale: boolean[] = []
  const decodeGate: { hold: boolean; waiting: (() => void)[] } = { hold: false, waiting: [] }

  const getBuffer = (path: string, signal: AbortSignal) =>
    new Promise<ArrayBuffer | null>((resolve, reject) => {
      const p: Pending = { path, signal, settled: false, resolve: (v) => { p.settled = true; resolve(v) }, reject: (e) => { p.settled = true; reject(e) } }
      pending.push(p)
      signal.addEventListener('abort', () => p.reject(new DOMException('aborted', 'AbortError')))
    })

  // a body is "decoded" to its own text; the text 'bad' is an invalid body
  const decode = async (layer: Layer, buf: ArrayBuffer): Promise<Frame | null> => {
    const text = new TextDecoder().decode(buf)
    if (decodeGate.hold) await new Promise<void>((r) => decodeGate.waiting.push(r))
    return text === 'bad' ? null : ({ layer, text } as unknown as Frame)
  }

  const session = createMapSession({
    getBuffer,
    decode,
    enabled: initialEnabled,
    onStatus: (s) => statuses.push(s),
    onFrame: (layer, frame) => frames.push({ layer, text: (frame as unknown as { text: string }).text }),
    discard: (layer, frame) => discarded.push({ layer, text: (frame as unknown as { text: string }).text }),
    onStale: (s) => stale.push(s),
    now: () => Date.now(),
  })

  const paths = () => pending.map((p) => p.path)
  const count = (path: string) => pending.filter((p) => p.path === path).length

  // Answers the oldest unanswered request for `path` (a string body, null for 503, or an Error to fail it).
  async function answer(path: string, body: string | null | Error) {
    const p = pending.find((q) => q.path === path && !q.settled)
    if (!p) throw new Error(`no pending request for ${path}; have ${paths().join(', ')}`)
    if (body instanceof Error) p.reject(body)
    else p.resolve(body === null ? null : bytes(body))
    await vi.advanceTimersByTimeAsync(0)
  }

  return {
    session, pending, frames, discarded, statuses, stale, paths, count, answer,
    holdDecode: (hold: boolean) => {
      decodeGate.hold = hold
      if (!hold) for (const r of decodeGate.waiting.splice(0)) r()
    },
  }
}

describe('createMapSession', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000_000)
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('polls the status at once, then once a second, and stops entirely on stop()', async () => {
    expect(STATUS_POLL_MS).toBe(1000)
    const h = harness()
    h.session.start()
    expect(h.paths()).toEqual(['/map'])
    await h.answer('/map', statusBody(1))
    await vi.advanceTimersByTimeAsync(999)
    expect(h.paths()).toEqual(['/map'])
    await vi.advanceTimersByTimeAsync(1)
    expect(h.paths()).toEqual(['/map', '/map'])
    h.session.stop()
    expect(h.pending[1].signal.aborted).toBe(true) // the in-flight poll is aborted
    await vi.advanceTimersByTimeAsync(10_000)
    expect(h.paths()).toEqual(['/map', '/map']) // and nothing polls again
    h.session.dispose()
  })

  it('does not start a second poll loop when start() is called twice', async () => {
    const h = harness()
    h.session.start()
    h.session.start()
    expect(h.count('/map')).toBe(1)
    h.session.dispose()
  })

  it('can be started again after a stop and polls at once', async () => {
    const h = harness()
    h.session.start()
    h.session.stop()
    h.session.start()
    expect(h.count('/map')).toBe(2)
    h.session.dispose()
  })

  it('fetches a changed layer once, then again only when its key changes', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', statusBody(1, { elevation: 3 }))
    expect(h.paths()).toEqual(['/map', '/map/elevation'])
    await h.answer('/map/elevation', 'e3')
    expect(h.frames).toEqual([{ layer: 'elevation', text: 'e3' }])

    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { elevation: 3 }))
    expect(h.count('/map/elevation')).toBe(1) // same epoch:seq, held

    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { elevation: 4 }))
    expect(h.count('/map/elevation')).toBe(2)
    await h.answer('/map/elevation', 'e4')
    expect(h.frames.map((f) => f.text)).toEqual(['e3', 'e4'])
    h.session.dispose()
  })

  it('refetches a layer after an epoch change even though its seq did not', async () => {
    const h = harness(only('grid'))
    h.session.start()
    await h.answer('/map', statusBody(1, { grid: 2 }))
    await h.answer('/map/grid', 'g-epoch1')
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(2, { grid: 2 }))
    expect(h.count('/map/grid')).toBe(2)
    await h.answer('/map/grid', 'g-epoch2')
    expect(h.frames.map((f) => f.text)).toEqual(['g-epoch1', 'g-epoch2'])
    h.session.dispose()
  })

  it('does not fetch disabled layers or layers with seq 0, and fetches a layer at once when it is enabled later', async () => {
    const h = harness(only('grid'))
    h.session.start()
    await h.answer('/map', statusBody(1, { grid: 1, cloud: 1, depth: 0 }))
    expect(h.paths()).toEqual(['/map', '/map/grid'])
    h.session.setEnabled(only('grid', 'cloud', 'depth'))
    expect(h.paths()).toEqual(['/map', '/map/grid', '/map/cloud']) // depth has nothing yet
    h.session.dispose()
  })

  it('keeps at most one request per layer in flight, then fetches the newer version as soon as the first lands', async () => {
    const h = harness(only('live'))
    h.session.start()
    await h.answer('/map', statusBody(1, { live: 1 }))
    expect(h.count('/map/live')).toBe(1)

    await vi.advanceTimersByTimeAsync(1000) // the live request is still unanswered
    await h.answer('/map', statusBody(1, { live: 2 }))
    expect(h.count('/map/live')).toBe(1)

    await h.answer('/map/live', 'live-1')
    expect(h.frames).toEqual([{ layer: 'live', text: 'live-1' }])
    expect(h.count('/map/live')).toBe(2) // the version seen while the first was in flight
    h.session.dispose()
  })

  it('marks a frame as held with the key of the status that triggered the fetch, not a newer one', async () => {
    const h = harness(only('live'))
    h.session.start()
    await h.answer('/map', statusBody(1, { live: 1 }))
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { live: 2 })) // seen while seq 1 is in flight
    await h.answer('/map/live', 'live-1') // lands, holds 1:1, so 1:2 is still due
    expect(h.count('/map/live')).toBe(2)
    await h.answer('/map/live', 'live-2')
    // now both are held: a further poll with seq 2 fetches nothing
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { live: 2 }))
    expect(h.count('/map/live')).toBe(2)
    h.session.dispose()
  })

  it('waits out the layer minimum interval and starts the fetch exactly when it has passed', async () => {
    const h = harness(only('cloud'))
    h.session.start()
    const t0 = Date.now()
    await h.answer('/map', statusBody(1, { cloud: 1 }))
    await h.answer('/map/cloud', 'c1')
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { cloud: 2 })) // 1000 ms after the first start: too early (cloud: 2000)
    expect(h.count('/map/cloud')).toBe(1)
    await vi.advanceTimersByTimeAsync(999)
    expect(h.count('/map/cloud')).toBe(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(h.count('/map/cloud')).toBe(2)
    expect(Date.now() - t0).toBe(2000)
    h.session.dispose()
  })

  it('leaves the previous frame in place and does not mark the layer held when the gateway answers 503', async () => {
    const h = harness(only('trajectory'))
    h.session.start()
    await h.answer('/map', statusBody(1, { trajectory: 1 }))
    await h.answer('/map/trajectory', 't1')
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, { trajectory: 2 }))
    await h.answer('/map/trajectory', null) // 503
    expect(h.frames).toEqual([{ layer: 'trajectory', text: 't1' }])
    await vi.advanceTimersByTimeAsync(1000) // still due: tried again
    expect(h.count('/map/trajectory')).toBeGreaterThanOrEqual(3)
    h.session.dispose()
  })

  it('treats a body the decoder rejects like a 503: no frame, not held, tried again', async () => {
    const h = harness(only('grid'))
    h.session.start()
    await h.answer('/map', statusBody(1, { grid: 1 }))
    await h.answer('/map/grid', 'bad')
    expect(h.frames).toEqual([])
    expect(h.discarded).toEqual([])
    await vi.advanceTimersByTimeAsync(1000)
    expect(h.count('/map/grid')).toBe(2)
    await h.answer('/map/grid', 'good')
    expect(h.frames).toEqual([{ layer: 'grid', text: 'good' }])
    h.session.dispose()
  })

  it('survives a failed layer request (gateway error) and retries it', async () => {
    const h = harness(only('depth'))
    h.session.start()
    await h.answer('/map', statusBody(1, { depth: 1 }))
    await h.answer('/map/depth', new ApiError({ title: 'Gateway unreachable', status: 0, detail: 'cannot reach /api/v1' }))
    expect(h.frames).toEqual([])
    await vi.advanceTimersByTimeAsync(500)
    expect(h.count('/map/depth')).toBe(2)
    h.session.dispose()
  })

  it('aborts in-flight layer fetches on stop() and treats the abort as no error, no frame', async () => {
    const h = harness(only('elevation', 'camera'))
    h.session.start()
    await h.answer('/map', statusBody(1, { elevation: 1, camera: 1 }))
    const inFlight = h.pending.filter((p) => p.path.startsWith('/map/'))
    expect(inFlight).toHaveLength(2)
    h.session.stop()
    await vi.advanceTimersByTimeAsync(0)
    expect(inFlight.every((p) => p.signal.aborted)).toBe(true)
    expect(h.frames).toEqual([])
    await vi.advanceTimersByTimeAsync(10_000)
    expect(h.pending).toHaveLength(3) // nothing restarts
    h.session.dispose()
  })

  it('drops a body that finishes decoding after stop(), handing the decoded frame back for disposal', async () => {
    const h = harness(only('camera'))
    h.holdDecode(true)
    h.session.start()
    await h.answer('/map', statusBody(1, { camera: 1 }))
    await h.answer('/map/camera', 'jpeg') // fetched, decode pending
    h.session.stop()
    h.holdDecode(false)
    await vi.advanceTimersByTimeAsync(0)
    expect(h.frames).toEqual([])
    expect(h.discarded).toEqual([{ layer: 'camera', text: 'jpeg' }])
    h.session.dispose()
  })

  it('reports status changes, but not a repeat of the same document', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', statusBody(1, {}, { keyframes: 1 }))
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, {}, { keyframes: 1 }))
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1, {}, { keyframes: 2 }))
    expect(h.statuses.map((s) => s.stats.keyframes)).toEqual([1, 2])
    h.session.dispose()
  })

  it('is stale after a failed poll and fresh again after a good one', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', new ApiError({ title: 'Gateway unreachable', status: 0, detail: '' }))
    expect(h.stale.at(-1)).toBe(true)
    await vi.advanceTimersByTimeAsync(1000)
    await h.answer('/map', statusBody(1))
    expect(h.stale.at(-1)).toBe(false)
    h.session.dispose()
  })

  it('counts an unreadable, mistyped or 503 status as a failed poll and fetches nothing from it', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', statusBody(1))
    expect(h.stale.at(-1)).toBe(false)
    for (const body of ['not json', '{"epoch":1}', null]) {
      await vi.advanceTimersByTimeAsync(1000)
      await h.answer('/map', body)
      expect(h.stale.at(-1)).toBe(true)
    }
    expect(h.paths().every((p) => p === '/map')).toBe(true)
    h.session.dispose()
  })

  it('turns stale on its own 3 s after the last successful poll, even when polling stopped', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', statusBody(1))
    expect(h.stale.at(-1)).toBe(false)
    h.session.stop()
    await vi.advanceTimersByTimeAsync(3000)
    expect(h.stale.at(-1)).toBe(false) // exactly 3000 ms is not "older than"
    await vi.advanceTimersByTimeAsync(2)
    expect(h.stale.at(-1)).toBe(true)
    h.session.dispose()
  })

  it('dispose() cancels the stale timer as well as the polling', async () => {
    const h = harness()
    h.session.start()
    await h.answer('/map', statusBody(1))
    const before = h.stale.length
    h.session.dispose()
    await vi.advanceTimersByTimeAsync(10_000)
    expect(h.stale).toHaveLength(before)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('keeps what it holds across stop() and start(): an unchanged layer is not fetched again', async () => {
    const h = harness(only('grid'))
    h.session.start()
    await h.answer('/map', statusBody(1, { grid: 1 }))
    await h.answer('/map/grid', 'g1')
    h.session.stop()
    h.session.start()
    await h.answer('/map', statusBody(1, { grid: 1 }))
    expect(h.count('/map/grid')).toBe(1)
    h.session.dispose()
  })
})
