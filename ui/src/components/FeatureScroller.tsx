import { useEffect, useRef, useState, type ReactNode } from 'react'
import MagnetLines from './ui/magnet-lines'

// Scroll-driven tour of what the operator console actually does. A pinned stage steps through the real
// features: the section is tall, scroll progress picks the active one, the others recede.
// The visuals are small illustrations, not live data.

const STEPS: { id: string; title: string; body: string; visual: ReactNode; caption: string }[] = [
  {
    id: 'health',
    title: 'The §12 health table',
    body: 'Camera, perception port, localization, TF, Nav2 heartbeat and e-stop, each with its age. A watch whose input stops arriving trips, it never keeps its last good value.',
    caption: 'six watches · fail closed',
    visual: (
      <div className="fs-ladder">
        <div>camera <b>ok</b></div>
        <div>perception <b>ok</b></div>
        <div>localization <b>trip</b></div>
        <div>nav2 <b>ok</b></div>
      </div>
    ),
  },
  {
    id: 'estop',
    title: 'E-stop',
    body: 'Level 1 of the safety precedence. The console asks the gateway to publish /ugv/e_stop, latched and repeated while asserted, so an arbiter that restarts still sees it.',
    caption: '/ugv/e_stop · latched',
    visual: (
      <div className="fs-chain">
        <span className="fs-node hot">console</span><span className="fs-arrow">→</span>
        <span className="fs-node">gateway</span><span className="fs-arrow">→</span>
        <span className="fs-node hot">/ugv/e_stop</span>
      </div>
    ),
  },
  {
    id: 'cmd',
    title: 'Final /cmd_vel',
    body: 'Shows what the safety authority actually let through to the base, read only. The console never publishes a drive command.',
    caption: '/cmd_vel · read only',
    visual: (
      <div className="fs-chain">
        <span className="fs-node">Nav2 candidate</span><span className="fs-arrow">→</span>
        <span className="fs-node hot">safety authority</span><span className="fs-arrow">→</span>
        <span className="fs-node">/cmd_vel</span>
      </div>
    ),
  },
  {
    id: 'goal',
    title: 'Mode and map-frame goal',
    body: 'Switch RTAB-Map between mapping and localize, and send a map-frame goal to Nav2. The gateway refuses a goal with the tripped watches as reasons while any §12 row is tripped.',
    caption: 'PUT mode · POST goal · 409 when held',
    visual: (
      <svg width="260" height="200" viewBox="0 0 260 200" role="img" aria-label="Goal in the map frame">
        <line x1="30" y1="170" x2="230" y2="170" stroke="#2b5f4c" />
        <line x1="30" y1="170" x2="30" y2="20" stroke="#2b5f4c" />
        <circle cx="70" cy="140" r="6" fill="#86f0cf" />
        <line x1="70" y1="140" x2="180" y2="60" stroke="#eafff7" strokeWidth="2" strokeDasharray="5 4" />
        <circle cx="180" cy="60" r="9" fill="none" stroke="#86f0cf" strokeWidth="2" />
        <text x="190" y="54" fill="#6c9f8c" fontSize="11" fontFamily="monospace">goal (map)</text>
      </svg>
    ),
  },
  {
    id: 'safe',
    title: 'Safety first',
    body: 'Precedence is fixed: e-stop, then health faults, then degraded perception or invalid pose, then the Nav2 candidate. The Dev 5 ugv_safety arbiter enforces it; this console only shows and asks.',
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
    <section ref={section} id="features" className="features" style={{ ['--steps' as string]: n }} aria-label="What the operator console does">
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
