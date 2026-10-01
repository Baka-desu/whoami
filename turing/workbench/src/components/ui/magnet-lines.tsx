import { useEffect, useRef } from 'react'
import { useCoarsePointer, useReducedMotion } from './use-motion-prefs'

interface Props {
  rows?: number
  cols?: number
  color?: string
  lineWidth?: number
  lineHeight?: number
  baseAngle?: number
  className?: string
}

// A field of short lines that swing to point at the cursor. Own implementation (not the
// @componentry package): one rAF-throttled pointer listener, transform-only updates, static at
// `baseAngle` for touch devices and reduced motion.
export default function MagnetLines({
  rows = 9, cols = 12, color = 'var(--accent)', lineWidth = 1.5, lineHeight = 22, baseAngle = -10, className = '',
}: Props) {
  const box = useRef<HTMLDivElement>(null)
  const reduced = useReducedMotion()
  const coarse = useCoarsePointer()
  const live = !reduced && !coarse

  useEffect(() => {
    const el = box.current
    if (!el || !live) return
    let raf = 0
    let ev: PointerEvent | null = null
    const apply = () => {
      raf = 0
      if (!ev) return
      for (const line of el.children as HTMLCollectionOf<HTMLElement>) {
        const r = line.getBoundingClientRect()
        const deg = (Math.atan2(ev.clientY - (r.top + r.height / 2), ev.clientX - (r.left + r.width / 2)) * 180) / Math.PI
        line.style.transform = `rotate(${deg - 90}deg)`
      }
    }
    const move = (e: PointerEvent) => {
      ev = e
      if (!raf) raf = requestAnimationFrame(apply)
    }
    window.addEventListener('pointermove', move, { passive: true })
    return () => {
      window.removeEventListener('pointermove', move)
      cancelAnimationFrame(raf)
    }
  }, [live])

  return (
    <div
      ref={box}
      aria-hidden="true"
      className={className}
      style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gridTemplateRows: `repeat(${rows}, 1fr)`, placeItems: 'center', pointerEvents: 'none' }}>
      {Array.from({ length: rows * cols }, (_, i) => (
        <span
          key={i}
          style={{ width: lineWidth, height: lineHeight, background: color, transform: `rotate(${baseAngle}deg)`, transition: 'transform 0.25s cubic-bezier(0.22,1,0.36,1)', willChange: live ? 'transform' : undefined }}
        />
      ))}
    </div>
  )
}
