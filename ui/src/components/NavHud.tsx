import type { Analysis } from '../types'

interface Instruction {
  icon: string
  text: string
  sub: string
  alert: boolean
}

const LOOKAHEAD_M = 2.5 // how far along the path we look to decide the next turn

// Turn-by-turn style instruction derived entirely from the already-computed local path - no
// planner/Nav2 involved, this is a display layer over the same {x,z} path the ground map draws.
// Distance shown is bounded by the analysis grid's forward extent (TH*CELL_M, ~12m), not the full
// mission - labelled "ahead", never implying a total-trip distance we don't have.
function computeInstruction(a: Analysis): Instruction {
  if (a.degraded) return { icon: '⏸', text: 'HOLD', sub: 'perception degraded', alert: true }
  if (a.path.length < 2) return { icon: '■', text: 'HOLD', sub: 'no safe path ahead', alert: true }

  const start = a.path[0]
  let dist = 0
  let target = a.path[a.path.length - 1]
  for (let i = 1; i < a.path.length; i++) {
    const prev = a.path[i - 1], cur = a.path[i]
    dist += Math.hypot(cur.x - prev.x, cur.z - prev.z)
    if (dist >= LOOKAHEAD_M) { target = cur; break }
  }
  const totalM = a.path.reduce((sum, p, i) => (i ? sum + Math.hypot(p.x - a.path[i - 1].x, p.z - a.path[i - 1].z) : 0), 0)

  const deg = (Math.atan2(target.x - start.x, target.z - start.z) * 180) / Math.PI
  const mag = Math.abs(deg)
  const dir = deg > 0 ? 'right' : 'left'
  const sub = `${totalM.toFixed(0)} m ahead`

  if (mag < 6) return { icon: '↑', text: 'Continue straight', sub, alert: false }
  if (mag < 20) return { icon: dir === 'right' ? '↗' : '↖', text: `Bear ${dir}`, sub, alert: false }
  if (mag < 45) return { icon: dir === 'right' ? '→' : '←', text: `Turn ${dir}`, sub, alert: false }
  return { icon: dir === 'right' ? '↘' : '↙', text: `Turn sharply ${dir}`, sub, alert: false }
}

export function NavHud({ analysis }: { analysis: Analysis | null }) {
  if (!analysis) return null
  const i = computeInstruction(analysis)
  return (
    <div className={`navhud ${i.alert ? 'alert' : ''}`}>
      <span className="navhud-icon">{i.icon}</span>
      <span className="navhud-text">
        <b>{i.text}</b>
        <i>{i.sub}</i>
      </span>
    </div>
  )
}
