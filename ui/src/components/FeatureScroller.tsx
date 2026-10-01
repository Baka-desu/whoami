import { useEffect, useRef, useState, type ReactNode } from 'react'
import MagnetLines from './ui/magnet-lines'

// Scroll-driven tour of what this workbench actually does. A pinned stage steps through the real
// features: the section is tall, scroll progress picks the active one, the others recede.
// The visuals are small illustrations, not live data.

const CELL = ['#2c4a40', '#8fd3ff', '#ff2a2a'] // unknown / traversable / hazard (same hues as the viewport)

function maskCells() {
  // fixed, hand-laid 16x7 pattern: hazard blob on the left, free ground in the middle, unknown at the far edge
  const rows = ['2200111111111000', '2200111111111100', '2000111111111100', '0001111111111100', '0011111111111110', '0111111111111110', '1111111111111110']
  return rows.join('').split('').map((c, i) => <i key={i} style={{ background: CELL[+c] }} />)
}

const STEPS: { id: string; title: string; body: string; visual: ReactNode; caption: string }[] = [
  {
    id: 'port',
    title: 'Perception port',
    body: 'Every frame is judged as a canonical mask: 0 unknown, 1 traversable, 2 hazard. Unknown is never free, and a mask older than 500 ms is stale and shown as such.',
    caption: 'mask {0,1,2} · freshness',
    visual: <div className="fs-cells">{maskCells()}</div>,
  },
  {
    id: 'loc',
    title: 'Localization and pose validity',
    body: 'The map → odom → base_link TF chain and the pose_valid heartbeat come from the localization stack. A heartbeat that stops reads NO SIGNAL, never its last value.',
    caption: 'tf · /ugv/pose_valid',
    visual: (
      <div className="fs-chain">
        <span className="fs-node">map</span><span className="fs-arrow">→</span>
        <span className="fs-node">odom</span><span className="fs-arrow">→</span>
        <span className="fs-node hot">base_link</span>
        <span className="fs-node hot" style={{ flexBasis: '100%', textAlign: 'center' }}>pose_valid</span>
      </div>
    ),
  },
  {
    id: 'nav',
    title: 'Nav2 costmap and plan',
    body: 'The local costmap and the planned path from Nav2 are drawn around the robot, heading up, so you see what the planner sees. The candidate velocity is shown, but it is only a candidate.',
    caption: 'costmap · /plan · /cmd_vel_nav2',
    visual: (
      <svg width="260" height="200" viewBox="0 0 260 200" role="img" aria-label="Plan around an obstacle">
        <rect width="260" height="200" fill="#0a2a22" stroke="#2b5f4c" />
        <rect x="150" y="60" width="44" height="50" fill="#ff2a2a" opacity="0.9" />
        <rect x="140" y="50" width="64" height="70" fill="#ffb36b" opacity="0.18" />
        <path d="M110 190 C110 140 100 110 120 70 S 130 20 130 8" stroke="#eafff7" strokeWidth="3" fill="none" />
        <path d="M130 190 l-8 -14 h16 z" fill="#86f0cf" transform="translate(-20,0)" />
      </svg>
    ),
  },
  {
    id: 'goal',
    title: 'Localized goal',
    body: 'Freeze a start pose, then give a distance and bearing or local x / y. The goal becomes one map-frame pose for Nav2, and it is blocked unless the pose is valid, Nav2 is alive, perception is healthy and e-stop is off.',
    caption: '50 m @ 20° from the start',
    visual: (
      <svg width="260" height="200" viewBox="0 0 260 200" role="img" aria-label="Start pose and goal at a bearing">
        <circle cx="70" cy="160" r="6" fill="#86f0cf" />
        <line x1="70" y1="160" x2="70" y2="40" stroke="#2b5f4c" strokeDasharray="4 4" />
        <line x1="70" y1="160" x2="152" y2="52" stroke="#eafff7" strokeWidth="2" />
        <path d="M70 100 A60 60 0 0 1 100 108" stroke="#f2e17c" fill="none" />
        <text x="104" y="104" fill="#f2e17c" fontSize="11" fontFamily="monospace">20°</text>
        <circle cx="152" cy="52" r="9" fill="none" stroke="#86f0cf" strokeWidth="2" />
        <text x="80" y="178" fill="#6c9f8c" fontSize="11" fontFamily="monospace">start</text>
        <text x="164" y="46" fill="#6c9f8c" fontSize="11" fontFamily="monospace">goal</text>
      </svg>
    ),
  },
  {
    id: 'safe',
    title: 'Safety first',
    body: 'Precedence is fixed: e-stop, then health faults, then degraded perception or invalid pose, then the Nav2 candidate. This UI can assert e-stop but never publishes a drive command.',
    caption: 'e-stop > health > degraded > nav2',
    visual: (
      <div className="fs-ladder">
        <div>1 e-stop <b>zero</b></div>
        <div>2 health fault <b>zero</b></div>
        <div>3 degraded / invalid pose <b>hold</b></div>
        <div>4 Nav2 candidate <b>forward</b></div>
      </div>
    ),
  },
]

export default function FeatureScroller() {
  const section = useRef<HTMLElement>(null)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    let raf = 0
    const update = () => {
      raf = 0
      const el = section.current
      if (!el) return
      const r = el.getBoundingClientRect()
      const span = r.height - window.innerHeight
      setProgress(span > 0 ? Math.min(1, Math.max(0, -r.top / span)) : 0)
    }
    const on = () => { if (!raf) raf = requestAnimationFrame(update) }
    update()
    window.addEventListener('scroll', on, { passive: true })
    window.addEventListener('resize', on)
    return () => {
      window.removeEventListener('scroll', on)
      window.removeEventListener('resize', on)
      cancelAnimationFrame(raf)
    }
  }, [])

  const n = STEPS.length
  const active = Math.min(n - 1, Math.floor(progress * n))

  const jump = (i: number) => {
    const el = section.current
    if (!el) return
    const span = el.offsetHeight - window.innerHeight
    window.scrollTo({ top: el.offsetTop + ((i + 0.5) / n) * span })
  }

  return (
    <section ref={section} id="features" className="features" style={{ ['--steps' as string]: n }} aria-label="What WHOAMI does">
      <div className="fs-stage">
        <div className="fs-list">
          <p className="fs-kicker">/ what it does</p>
          {STEPS.map((s, i) => (
            <button
              key={s.id}
              type="button"
              className={`fs-item ${i === active ? 'active' : i < active ? 'past' : ''}`}
              aria-current={i === active}
              onClick={() => jump(i)}>
              <span className="n">0{i + 1}</span>
              <h4>{s.title}</h4>
              <p>{s.body}</p>
            </button>
          ))}
        </div>
        <div className="fs-visual">
          <MagnetLines className="fs-magnet" rows={9} cols={13} />
          {STEPS.map((s, i) => (
            <div key={s.id} className={`fs-slide ${i === active ? 'active' : i < active ? 'past' : ''}`} aria-hidden={i !== active}>
              {s.visual}
              <p className="fs-cap">{s.caption} · illustration</p>
            </div>
          ))}
          <div className="fs-progress"><i style={{ transform: `scaleX(${progress})` }} /></div>
        </div>
      </div>
    </section>
  )
}
