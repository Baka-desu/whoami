import { useState } from 'react'

type Mode = 'bearing' | 'coords'

interface Props {
  connected: boolean
  status: string
  onSend: (x: number, y: number, yawRad: number) => void
}

// V1 "localized goal" (architecture addendum): a destination relative to the start pose, e.g.
// "50 m at 20 deg east of north" - no GPS, the destination need not be visible beforehand. Bearing
// is standard compass convention (0=N, 90=E, clockwise) converted to the map frame's ENU axes
// (x=East, y=North, REP-103); the robot is pointed along the travel direction on arrival.
export function GoalPanel({ connected, status, onSend }: Props) {
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
        disabled={!connected || !valid}
        onClick={() => onSend(goalX, goalY, yawRad)}
        style={{ marginTop: 10, width: '100%' }}
      >
        SEND GOAL
      </button>
      {!connected && <p className="dim">Connect to ROS 2 first — this calls /navigate_to_pose on that connection.</p>}
      {status && <p className="dim">{status}</p>}
    </section>
  )
}
