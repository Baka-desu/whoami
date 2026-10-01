import { useEffect, useRef, useState, type ReactNode } from 'react'
import MagnetLines from './ui/magnet-lines'

// Scroll-driven tour of what the Dev 1 eval workbench actually does. A pinned stage steps through the real
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
    id: 'depth',
    title: 'Metric depth',
    body: 'When the Depth Anything 3 Metric Large channel is running, its 32FC1 depth for the same image stamp is shown beside the mask. A depth image from another frame is never paired with this mask.',
    caption: '/perception/depth/image',
    visual: (
      <svg width="260" height="200" viewBox="0 0 260 200" role="img" aria-label="Depth gradient">
        <defs><linearGradient id="fsd" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stopColor="#d9902d" /><stop offset="1" stopColor="#170f04" /></linearGradient></defs>
        <rect width="260" height="200" fill="url(#fsd)" stroke="#2b5f4c" />
      </svg>
    ),
  },
  {
    id: 'sources',
    title: 'Eval sources',
    body: 'Feed a recorded bag through rosbridge, or look at a single photo or webcam frame. Photos and the webcam have no perception backend and no CameraInfo, so they say NO ANALYZER and K ASSUMED instead of faking a result.',
    caption: 'bag · photo · webcam',
    visual: (
      <div className="fs-chain">
        <span className="fs-node hot">bag</span><span className="fs-arrow">→</span>
        <span className="fs-node">Dev 1 port</span><span className="fs-arrow">→</span>
        <span className="fs-node">workbench</span>
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
    <section ref={section} id="features" className="features" style={{ ['--steps' as string]: n }} aria-label="What the perception workbench does">
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
