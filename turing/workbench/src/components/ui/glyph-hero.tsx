import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { useCoarsePointer, useReducedMotion } from './use-motion-prefs'

// three.js is large, so the ring is its own lazily loaded chunk, and only mounts while it is on
// screen and the tab is visible (no GPU work in the background).
const GlyphRing = lazy(() => import('./glyph-ring'))

export default function GlyphHero({ className = '' }: { className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [onScreen, setOnScreen] = useState(false)
  const [tabVisible, setTabVisible] = useState(() => !document.hidden)
  const reduced = useReducedMotion()
  const coarse = useCoarsePointer()

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(([e]) => setOnScreen(e.isIntersecting))
    io.observe(el)
    const vis = () => setTabVisible(!document.hidden)
    document.addEventListener('visibilitychange', vis)
    return () => {
      io.disconnect()
      document.removeEventListener('visibilitychange', vis)
    }
  }, [])

  return (
    <div ref={ref} className={className} aria-hidden="true">
      {!reduced && onScreen && tabVisible && (
        <Suspense fallback={null}>
          <GlyphRing rings={coarse ? 9 : 18} scale={coarse ? 140 : 200} />
        </Suspense>
      )}
    </div>
  )
}
