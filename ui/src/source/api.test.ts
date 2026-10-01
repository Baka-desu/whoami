import { describe, expect, it } from 'vitest'
import {
  API_BASE, TELEMETRY_MAX_AGE_MS, WATCH_NAMES, applyEvent, asCommand, asGoal, asNavigation, asProblem, asSafety,
  isLive,
} from './api'

// Payloads in the exact shape ugv_api's schemas.py serialises (camelCase).
const watches = (trip?: string) =>
  WATCH_NAMES.map((name) => ({ name, ok: name !== trip, reason: name === trip ? 'stale (0.90 s > 0.50 s)' : 'ok', ageS: 0.05 }))

const safety = (trip?: string) => ({
  ok: !trip,
  watches: watches(trip),
  arbiter: { status: null, ageS: null, present: false },
  eStop: { asserted: false, assertedByGateway: false, lastSeen: null, lastSeenAgeS: null },
})

const goal = { id: 'abc', state: 'executing', frameId: 'map', x: 3, y: 0, yaw: 0, distanceRemaining: 2.5, recoveries: 0, errorCode: null, errorMessage: null }

describe('gateway payload checks', () => {
  it('accepts a healthy §12 table', () => {
    const s = asSafety(safety())
    expect(s?.ok).toBe(true)
    expect(s?.watches.map((w) => w.name)).toEqual([...WATCH_NAMES])
  })

  it('never trusts an ok summary that contradicts a tripped row', () => {
    const forged = { ...safety('nav2'), ok: true }
    expect(asSafety(forged)?.ok).toBe(false)
  })

  it('rejects a table missing a §12 row or with an unknown row', () => {
    const missing = { ...safety(), watches: watches().filter((w) => w.name !== 'tf') }
    expect(asSafety(missing)).toBeNull()
    const unknown = { ...safety(), watches: [...watches(), { name: 'gps', ok: true, reason: 'ok', ageS: 0 }] }
    expect(asSafety(unknown)).toBeNull()
  })

  it('checks /cmd_vel consistency', () => {
    expect(asCommand({ available: false, linear: null, angular: null, ageS: null })?.available).toBe(false)
    expect(asCommand({ available: true, linear: null, angular: null, ageS: 0.1 })).toBeNull()
    expect(asCommand({ available: true, linear: { x: 0.2, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0.1 }, ageS: 0.1 })).not.toBeNull()
  })

  it('validates goals and navigation', () => {
    expect(asGoal(goal)?.id).toBe('abc')
    expect(asGoal({ ...goal, state: 'flying' })).toBeNull()
    expect(asGoal({ ...goal, x: 'north' })).toBeNull()
    const nav = { heartbeat: true, heartbeatAgeS: 0.02, status: 'ok', actionServerReady: true, activeGoal: goal }
    expect(asNavigation(nav)?.activeGoal?.id).toBe('abc')
    expect(asNavigation({ ...nav, activeGoal: { id: 1 } })).toBeNull()
  })

  it('reads RFC 9457 problems, including goal-gate reasons', () => {
    const p = asProblem({ type: 'about:blank', title: 'Goal gate closed', status: 409, detail: 'x', reasons: ['e_stop: asserted'] }, 409)
    expect(p).toMatchObject({ title: 'Goal gate closed', status: 409, reasons: ['e_stop: asserted'] })
    expect(asProblem('<html>', 502)).toEqual({ title: 'HTTP 502', status: 502, detail: '' })
  })
})

describe('telemetry stream', () => {
  it('applies valid events and ignores malformed ones', () => {
    const t1 = applyEvent(null, 'safety', JSON.stringify(safety()), 1000)
    expect(t1?.safety?.ok).toBe(true)
    expect(applyEvent(t1, 'safety', '{"ok": true}', 1100)).toBe(t1)
    expect(applyEvent(t1, 'safety', 'not json', 1100)).toBe(t1)
    expect(applyEvent(t1, 'unknown', '{}', 1100)).toBe(t1)
  })

  it('goes NO SIGNAL once the stream stops', () => {
    const t = applyEvent(null, 'safety', JSON.stringify(safety()), 1000)
    expect(isLive(t, 1000 + TELEMETRY_MAX_AGE_MS)).toBe(true)
    expect(isLive(t, 1001 + TELEMETRY_MAX_AGE_MS)).toBe(false)
    expect(isLive(null, 0)).toBe(false)
  })
})

describe('operator boundary', () => {
  const sources = import.meta.glob('/src/**/*.{ts,tsx}', { query: '?raw', import: 'default', eager: true }) as Record<string, string>
  const production = Object.entries(sources).filter(([path]) => !/\.test\.tsx?$/.test(path))
  // The camera view (the main page, its rosbridge reader, source picker and perception widgets) is display-only,
  // by the owner's call.
  const LIVE_VIEW = /\/src\/(components\/(CameraView|Viewport|TopDownMap|Inspector|SourcePanel)\.tsx|source\/(rosbridge|rosimage|camera|useCameraSource)\.ts|analysis\/.*|types\.ts)$/

  it('never commands motion and never publishes to ROS', () => {
    expect(API_BASE).toBe('/api/v1')
    const forbidden = [/op:\s*['"]publish/, /op:\s*['"](advertise|call_service|send_action_goal)/, /cmd_vel_nav2/]
    const offenders = production.flatMap(([path, code]) => forbidden.filter((re) => re.test(code)).map((re) => `${path}: ${re}`))
    expect(offenders).toEqual([])
  })

  it('controls go only through the gateway: rosbridge is confined to the read-only live view', () => {
    const forbidden = [/new WebSocket/, /rosbridge/i, /['"]\/segmentation\//, /['"]\/perception\//, /['"]\/odom['"]/, /['"]\/tf['"]/,
      /['"]\/plan['"]/, /costmap/i]
    const offenders = production
      .filter(([path]) => !LIVE_VIEW.test(path))
      .flatMap(([path, code]) => forbidden.filter((re) => re.test(code)).map((re) => `${path}: ${re}`))
    expect(offenders).toEqual([])
  })
})
