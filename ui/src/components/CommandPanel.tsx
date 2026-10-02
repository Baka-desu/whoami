import { useState } from 'react'
import type { Goal, Mode } from '../source/api'

interface Props {
  live: boolean
  estopAsserted: boolean
  estopBusy: boolean
  onEstop: (asserted: boolean) => void
  mode: Mode | null
  onMode: (mode: Mode) => void
  gateReasons: string[]
  activeGoal: Goal | null
  onSendGoal: (x: number, y: number, yawRad: number) => void
  onCancelGoal: (id: string) => void
  message: { text: string; reasons?: string[]; error: boolean } | null
}

const CANCELABLE = new Set(['executing', 'canceling'])

// The operator's three commands (architecture.md §3.1 e-stop, §10 mode, §11 map-frame goal), all sent to
// Dev 5's gateway. The gateway re-checks the goal gate and answers 409 with reasons; this panel only
// pre-disables the button.
export function CommandPanel(p: Props) {
  const [x, setX] = useState('0')
  const [y, setY] = useState('0')
  const [yawDeg, setYawDeg] = useState('0')
  const gx = Number(x), gy = Number(y), gyaw = Number(yawDeg)
  const valid = x.trim() !== '' && y.trim() !== '' && yawDeg.trim() !== ''
    && Number.isFinite(gx) && Number.isFinite(gy) && Number.isFinite(gyaw)
  const blocked = !p.live ? ['no telemetry from the gateway'] : p.gateReasons

  return (
    <>
      <section className="section">
        <h3>E-stop</h3>
        <button
          className={`btn danger estop ${p.estopAsserted ? 'primary' : ''}`}
          disabled={p.estopBusy}
          onClick={() => p.onEstop(!p.estopAsserted)}
        >
          {p.estopAsserted ? 'RELEASE E-STOP' : 'E-STOP'}
        </button>
        <p className="dim">
          Level 1 operator kill (§3.1): the gateway publishes /ugv/e_stop, latched and repeated while asserted.
          If the gateway is unreachable, use the CLI: <code>ros2 topic pub /ugv/e_stop std_msgs/msg/Bool "{'{data: true}'}"</code>
        </p>
      </section>

      <section className="section">
        <h3>Localization mode</h3>
        <div className="tabs two">
          {(['mapping', 'localize'] as const).map((m) => (
            <button key={m} className={p.mode === m ? 'on' : ''} disabled={!p.live} onClick={() => p.onMode(m)}>
              {m.toUpperCase()}
            </button>
          ))}
        </div>
        <p className="dim">§10: mapping builds and saves the RTAB-Map database; localize navigates on it.
          {p.mode === null && ' No mode has been set from this console yet.'}</p>
      </section>

      <section className="section">
        <h3>Goal (map frame)</h3>
        <div className="stack">
          <label>x (m)<input inputMode="decimal" value={x} onChange={(e) => setX(e.target.value)} /></label>
          <label>y (m)<input inputMode="decimal" value={y} onChange={(e) => setY(e.target.value)} /></label>
          <label>yaw (°, CCW from +x)<input inputMode="decimal" value={yawDeg} onChange={(e) => setYawDeg(e.target.value)} /></label>
        </div>
        <button
          className="btn primary"
          disabled={!valid || blocked.length > 0}
          onClick={() => p.onSendGoal(gx, gy, (gyaw * Math.PI) / 180)}
          style={{ marginTop: 12, width: '100%' }}
        >
          SEND GOAL
        </button>
        <button
          className="btn"
          disabled={!p.activeGoal || !CANCELABLE.has(p.activeGoal.state)}
          onClick={() => p.activeGoal && p.onCancelGoal(p.activeGoal.id)}
          style={{ marginTop: 8, width: '100%' }}
        >
          CANCEL GOAL
        </button>
        {!valid && <p className="dim">Enter finite x, y and yaw.</p>}
        {blocked.length > 0 && (
          <>
            <p className="dim">Goal gate closed:</p>
            {blocked.map((r) => <p className="reason" key={r}>{r}</p>)}
          </>
        )}
        <p className="dim">Sent to Nav2 /navigate_to_pose through the gateway. This console never publishes /cmd_vel.</p>
      </section>

      {p.message && (
        <section className="section">
          <h3>{p.message.error ? 'Refused' : 'Result'}</h3>
          <p className={p.message.error ? 'reason' : 'dim'}>{p.message.text}</p>
          {p.message.reasons?.map((r) => <p className="reason" key={r}>{r}</p>)}
        </section>
      )}
    </>
  )
}
