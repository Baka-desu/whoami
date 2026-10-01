// One shared "is this result still trustworthy" clock, so the viewport, the inspector and the
// ground map can never disagree. A frozen live stream must go stale on its own; a still (upload /
// take-photo) has nothing to go stale against, so it stays "captured" forever.
import { useEffect, useState } from 'react'
import { PERCEPTION_MAX_AGE_MS, type Analysis } from '../types'

export interface Freshness {
  ageMs: number
  stale: boolean // streaming only: age past perception_max_age
  degraded: boolean // the analysis itself flagged a problem (from a.degraded / a.reasons)
  ok: boolean // safe to present as a current, trustworthy result
  label: string // short status word for badges
}

const TICK_MS = 200

export function useFreshness(a: Analysis | null): Freshness | null {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!a?.meta.streaming) return
    const id = window.setInterval(() => setNow(Date.now()), TICK_MS)
    return () => window.clearInterval(id)
  }, [a?.meta.streaming, a?.meta.receivedAt, a?.meta.stamp])

  if (!a) return null
  if (!a.meta.streaming) {
    return { ageMs: a.ageMs, stale: false, degraded: a.degraded, ok: !a.degraded, label: a.degraded ? 'DEGRADED' : 'CAPTURED' }
  }
  const baseTime = a.meta.receivedAt ?? a.meta.stamp
  const ageMs = Math.max(0, now - baseTime)
  const stale = ageMs > PERCEPTION_MAX_AGE_MS
  const degraded = a.degraded
  return {
    ageMs, stale, degraded, ok: !stale && !degraded,
    label: stale ? 'STALE' : degraded ? 'DEGRADED' : 'LIVE',
  }
}
