import { useState } from 'react'

type Mode = 'bearing' | 'coords'

interface Props {
  connected: boolean
  status: string
  // fwd/left in metres in the start pose's frame (x forward, y left, REP-103), relYaw relative to its heading
  onSend: (fwd: number, left: number, relYawRad: number) => void
  // Gates sending on Dev 2's pose-valid heartbeat and the shared freshness clock, not just the
  // socket being open (archV1.md §9/§11: no Nav2 use of the reference pose without a valid one).
  canSend: boolean
  blockedReason: string
  hasStart: boolean
  canSetStart: boolean
  onSetStart: () => void
}

// V1 "localized goal" (archV1.md §9): a destination relative to a mission reference pose, e.g.
// "50 m at 20 deg from the starting pose" - no GPS, the destination need not be visible
// beforehand. The reference pose is frozen from Dev 2's live map->base_link TF (SET START, only
// while /ugv/pose_valid is true); the goal is then converted into a single `map`-frame
// PoseStamped and sent to Nav2's /navigate_to_pose. Bearing is clockwise from the start heading;
// coordinates are x forward / y left in the start pose's frame.
//
// Dev 2 exposes no mission service (interfaces.md), so the freeze + conversion happen in this UI,
// using only Dev 2's published outputs (/tf, /ugv/pose_valid). If Dev 2 later adds a reference-pose
// service, replace setStart/convert in App.tsx with a call to it.
export function GoalPanel({ connected, status, onSend, canSend, blockedReason, hasStart, canSetStart, onSetStart }: Props) {
  const [mode, setMode] = useState<Mode>('bearing')
  const [distance, setDistance] = useState('50')
  const [bearing, setBearing] = useState('20')
  const [x, setX] = useState('0')
  const [y, setY] = useState('0')

  const d = Number(distance), b = Number(bearing), gx = Number(x), gy = Number(y)
  const valid = mode === 'bearing' ? Number.isFinite(d) && d > 0 && Number.isFinite(b) : Number.isFinite(gx) && Number.isFinite(gy)

  let fwd = 0, left = 0, relYaw = 0
  if (mode === 'bearing' && valid) {
    const rad = (b * Math.PI) / 180 // clockwise from start heading
    fwd = d * Math.cos(rad)
    left = -d * Math.sin(rad)
    relYaw = -rad
  } else if (valid) {
    fwd = gx
    left = gy
    relYaw = Math.atan2(gy, gx)
  }

  return (
    <section className="section">
      <h3>Goal</h3>
      <button className="btn" disabled={!canSetStart} onClick={onSetStart} style={{ width: '100%', marginBottom: 12 }}>
        {hasStart ? 'RESET START POSE' : 'SET START POSE'}
      </button>
      <p className="dim" style={{ marginTop: 0, marginBottom: 12 }}>
        {hasStart ? 'Start pose frozen in map frame.' : 'Freeze the reference pose first (needs a valid pose).'}
      </p>

      <div className="tabs">
        <button className={mode === 'bearing' ? 'on' : ''} onClick={() => setMode('bearing')}>DISTANCE + BEARING</button>
        <button className={mode === 'coords' ? 'on' : ''} onClick={() => setMode('coords')}>LOCAL X / Y</button>
      </div>

      {mode === 'bearing' ? (
        <div className="stack">
          <label>Distance from start (m)
            <input inputMode="decimal" value={distance} onChange={(e) => setDistance(e.target.value)} />
          </label>
          <label>Bearing (° clockwise from start heading)
            <input inputMode="decimal" value={bearing} onChange={(e) => setBearing(e.target.value)} />
          </label>
        </div>
      ) : (
        <div className="stack">
          <label>X, forward of start (m)
            <input inputMode="decimal" value={x} onChange={(e) => setX(e.target.value)} />
          </label>
          <label>Y, left of start (m)
            <input inputMode="decimal" value={y} onChange={(e) => setY(e.target.value)} />
          </label>
        </div>
      )}

      <p className="dim">
        {valid ? `→ start frame: ${fwd.toFixed(1)} m forward, ${left.toFixed(1)} m left` : 'enter a valid distance/bearing or x/y'}
      </p>

      <button
        className="btn primary"
        disabled={!connected || !valid || !canSend}
        onClick={() => onSend(fwd, left, relYaw)}
        style={{ marginTop: 10, width: '100%' }}
      >
        SEND GOAL
      </button>
      {!connected && <p className="dim">Connect to ROS 2 first — this calls /navigate_to_pose on that connection.</p>}
      {connected && !hasStart && <p className="dim">Set the start pose before sending a goal.</p>}
      {connected && blockedReason && <p className="dim">Blocked: {blockedReason} — sending a goal now could drive on an unsafe reference pose.</p>}
      {status && <p className="dim">{status}</p>}
    </section>
  )
}
