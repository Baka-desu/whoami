import { useState } from 'react'

type Mode = 'bearing' | 'coords'

interface Props {
  connected: boolean
  status: string
  onSend: (x: number, y: number, yawRad: number) => void
  // Gates sending on Dev 2's pose-valid heartbeat and the shared freshness clock, not just the
  // socket being open (archV1.md §9/§11: no Nav2 use of the reference pose without a valid one).
  canSend: boolean
  blockedReason: string
}

// V1 "localized goal" (archV1.md): a destination relative to the start pose, e.g. "50 m at 20 deg
// east of north" - no GPS, the destination need not be visible beforehand. Bearing is standard
// compass convention (0=N, 90=E, clockwise) converted to the map frame's ENU axes (x=East,
// y=North, REP-103); the robot is pointed along the travel direction on arrival.
//
// Known gap vs archV1.md §5.1/§9: the {range,bearing}->map-frame conversion is owned by Dev 2,
// who freezes a reference pose in `map` at mission start and does the conversion there - Dev 2
// does not consume the mask and does not call Nav2. There is no rosbridge topic/service in
// interfaces.md yet for the UI to hand a raw {range,bearing} mission to that conversion, so this
// panel does the conversion itself and sends a PoseStamped goal directly. That is a UI-side
// stand-in, not the archV1.md-conformant path; canSend/blockedReason (Dev 2's /ugv/pose_valid +
// the shared freshness clock) is the safety net this panel *can* enforce without inventing a new
// cross-team contract.
export function GoalPanel({ connected, status, onSend, canSend, blockedReason }: Props) {
  const [mode, setMode] = useState<Mode>('bearing')
  const [distance, setDistance] = useState('50')
  const [bearing, setBearing] = useState('20')
  const [x, setX] = useState('0')
  const [y, setY] = useState('0')

  const d = Number(distance), b = Number(bearing), gx = Number(x), gy = Number(y)
  const valid = mode === 'bearing' ? Number.isFinite(d) && d > 0 && Number.isFinite(b) : Number.isFinite(gx) && Number.isFinite(gy)

  let goalX = 0, goalY = 0, yawRad = 0
  if (mode === 'bearing' && valid) {
    const rad = ((90 - b) * Math.PI) / 180 // compass bearing -> ENU angle from +X (East)
    goalX = d * Math.cos(rad)
    goalY = d * Math.sin(rad)
    yawRad = rad
  } else if (valid) {
    goalX = gx
    goalY = gy
    yawRad = Math.atan2(gy, gx)
  }

  return (
    <section className="section">
      <h3>Goal</h3>
      <div className="tabs">
        <button className={mode === 'bearing' ? 'on' : ''} onClick={() => setMode('bearing')}>DISTANCE + BEARING</button>
        <button className={mode === 'coords' ? 'on' : ''} onClick={() => setMode('coords')}>LOCAL X / Y</button>
      </div>

      {mode === 'bearing' ? (
        <div className="stack">
          <label>Distance from start (m)
            <input inputMode="decimal" value={distance} onChange={(e) => setDistance(e.target.value)} />
          </label>
          <label>Bearing (° from North, clockwise — 90 = East)
            <input inputMode="decimal" value={bearing} onChange={(e) => setBearing(e.target.value)} />
          </label>
        </div>
      ) : (
        <div className="stack">
          <label>X, east of start (m)
            <input inputMode="decimal" value={x} onChange={(e) => setX(e.target.value)} />
          </label>
          <label>Y, north of start (m)
            <input inputMode="decimal" value={y} onChange={(e) => setY(e.target.value)} />
          </label>
        </div>
      )}

      <p className="dim">
        {valid ? `→ map frame x=${goalX.toFixed(1)} m, y=${goalY.toFixed(1)} m` : 'enter a valid distance/bearing or x/y'}
      </p>

      <button
        className="btn primary"
        disabled={!connected || !valid || !canSend}
        onClick={() => onSend(goalX, goalY, yawRad)}
        style={{ marginTop: 10, width: '100%' }}
      >
        SEND GOAL
      </button>
      {!connected && <p className="dim">Connect to ROS 2 first — this calls /navigate_to_pose on that connection.</p>}
      {connected && blockedReason && <p className="dim">Blocked: {blockedReason} — sending a goal now could drive on an unsafe reference pose.</p>}
      {status && <p className="dim">{status}</p>}
    </section>
  )
}
