import type { ReactNode } from 'react'
import { fresh, type RobotSnapshot } from '../source/robot'
import { NavMap } from './NavMap'

function Row({ k, v, tone }: { k: string; v: ReactNode; tone?: 'trip' | 'warn' }) {
  return (
    <div className={`row ${tone ?? ''}`}>
      <span>{k}</span>
      <b>{v}</b>
    </div>
  )
}

const bool = (v: boolean | undefined, yes = 'TRUE', no = 'FALSE') => (v === undefined ? 'NO SIGNAL' : v ? yes : no)

// Live robot state from ugv_nav (Dev 2) and ugv_navigation (Dev 4). A heartbeat that stops
// arriving reads NO SIGNAL (fail-closed), never its last value.
export function RobotPanel({ robot: r, estop }: { robot: RobotSnapshot; estop: boolean }) {
  const pose = fresh(r.poseValid, r.now)
  const hb = fresh(r.nav2Heartbeat, r.now)
  const pd = fresh(r.perceptionDegraded, r.now)
  const odom = fresh(r.odom, r.now, 2000)
  const cmd = fresh(r.cmdVelNav2, r.now, 2000)
  const loc = r.locStatus?.value
  const nav2 = r.nav2Status?.value

  return (
    <>
      <section className="section">
        <h3>Navigation map</h3>
        <NavMap robot={r} />
        <div className="mini-legend">
          <span><i style={{ background: '#153a4d' }} />FREE</span>
          <span><i style={{ background: '#2c2c2c' }} />UNKNOWN</span>
          <span><i style={{ background: '#ff2a2a' }} />LETHAL</span>
          <span><i style={{ background: '#fff' }} />PLAN</span>
        </div>
        <p className="dim">Nav2 local costmap, robot heading up.</p>
      </section>

      <section className="section">
        <h3>Robot</h3>
        <Row k="e-stop (sent)" v={estop ? 'ASSERTED' : 'released'} tone={estop ? 'trip' : undefined} />
        <Row k="pose_valid" v={bool(pose)} tone={pose === false ? 'trip' : pose === undefined ? 'warn' : undefined} />
        {loc && loc !== 'valid' && <p className="reason">{loc.split(',').join(' · ')}</p>}
        <Row k="odom source" v={fresh(r.odomSource, r.now, Infinity)?.toUpperCase() ?? 'NO SIGNAL'} />
        <Row k="nav2_heartbeat" v={bool(hb, 'ACTIVE', 'DOWN')} tone={hb === false ? 'trip' : hb === undefined ? 'warn' : undefined} />
        {nav2 && nav2 !== 'ok' && <p className="reason">{nav2}</p>}
        <Row k="perception_degraded" v={bool(pd)} tone={pd ? 'trip' : undefined} />
        <Row k="speed (odom)" v={odom ? `${odom.v.toFixed(2)} m/s · ${odom.w.toFixed(2)} rad/s` : 'NO SIGNAL'} />
        <Row k="cmd_vel_nav2" v={cmd ? `${cmd.v.toFixed(2)} m/s · ${cmd.w.toFixed(2)} rad/s` : 'idle'} />
        <p className="dim">Candidate only; the safety authority owns /cmd_vel.</p>
      </section>
    </>
  )
}
