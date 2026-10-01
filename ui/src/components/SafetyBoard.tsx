import type { SafetyStatus, WatchName } from '../source/api'
import GlyphHero from './ui/glyph-hero'

// architecture.md §12: the minimum timeout table. Any tripped row means the safety authority must hold.
const ROW_LABEL: Record<WatchName, string> = {
  camera: 'Camera',
  perception: 'Perception port',
  localization: 'Localization',
  tf: 'TF map → base_link',
  nav2: 'Nav2 heartbeat',
  e_stop: 'E-stop',
}

const age = (s: number | null) => (s === null ? '—' : s < 10 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(0)} s`)

export function SafetyBoard({ safety, live }: { safety: SafetyStatus | undefined; live: boolean }) {
  if (!live || !safety) {
    return (
      <main className="viewport board">
        <div className="nosignal">
          <GlyphHero className="glyph-bg" />
          <b>NO SIGNAL</b>
          <span>No telemetry from the operator gateway (/api/v1). Treat the robot as unsupervised.</span>
        </div>
      </main>
    )
  }
  const tripped = safety.watches.filter((w) => !w.ok).length
  return (
    <main className="viewport board">
      <div className="board-inner">
        <div className={`verdict ${safety.ok ? 'ok' : 'hold'}`}>
          <span>§12 HEALTH</span>
          <b>{safety.ok ? 'ALL CLEAR' : `HOLD · ${tripped} TRIPPED`}</b>
          <em>{safety.ok ? 'no watch has tripped' : 'any trip must zero /cmd_vel'}</em>
        </div>
        <div className="watch-grid">
          {safety.watches.map((w) => (
            <div key={w.name} className={`watch ${w.ok ? 'ok' : 'trip'}`}>
              <span>{ROW_LABEL[w.name]}</span>
              <b>{w.ok ? 'OK' : 'TRIP'}</b>
              <p>{w.ok ? `age ${age(w.ageS)}` : w.reason}</p>
            </div>
          ))}
        </div>
        <div className={`arbiter ${safety.arbiter.present ? '' : 'absent'}`}>
          <span>Safety arbiter (Dev 5)</span>
          <b>{safety.arbiter.present ? safety.arbiter.status : 'NOT RUNNING'}</b>
          <p>
            {safety.arbiter.present
              ? `/ugv/safety_status · ${age(safety.arbiter.ageS)} ago`
              : 'Nothing publishes /ugv/safety_status, so nothing is gating /cmd_vel. This table is the operator view only.'}
          </p>
        </div>
      </div>
    </main>
  )
}
