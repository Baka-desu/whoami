import type { ReactNode } from 'react'
import type { Freshness } from '../analysis/freshness'
import { PERCEPTION_MAX_AGE_MS, CLASS_NAMES, CLASS_RGB, type Analysis } from '../types'
import { TopDownMap } from './TopDownMap'
import DraggableWidgetGrid, { type WidgetItem, type WidgetSize } from './ui/draggable-widget-grid'

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="section wcontent">
      <h3>{title}</h3>
      {children}
    </section>
  )
}

function Row({ k, v, tone }: { k: string; v: ReactNode; tone?: 'trip' | 'warn' }) {
  return (
    <div className={`row ${tone ?? ''}`}>
      <span>{k}</span>
      <b>{v}</b>
    </div>
  )
}

interface Props {
  analysis: Analysis | null
  freshness: Freshness | null
}

type WidgetId = 'analysis' | 'ground' | 'classes' | 'depth' | 'health' | 'source'

const SIZE: Record<WidgetId, WidgetSize> = {
  analysis: 'wide', ground: 'lg', classes: 'wide', depth: 'wide', health: 'wide', source: 'wide',
}
const LABEL: Record<WidgetId, string> = {
  analysis: 'Analysis', ground: 'Ground map',
  classes: 'Classes', depth: 'Depth', health: 'Health', source: 'Source',
}
const ORDER_KEY = 'ugv.console.perception.order'

// The arrangement is a per-browser preference (not app state): saved order, filtered to the widgets
// that exist right now.
function ordered(ids: WidgetId[]): WidgetItem[] {
  let saved: string[] = []
  try { saved = JSON.parse(localStorage.getItem(ORDER_KEY) ?? '[]') } catch { /* ignore */ }
  const rank = (id: string) => { const i = saved.indexOf(id); return i < 0 ? 999 : i }
  return [...ids]
    .sort((x, y) => rank(x) - rank(y))
    .map((id) => ({ id, size: SIZE[id], label: LABEL[id] }))
}

function remember(items: WidgetItem[]) {
  try {
    const saved: string[] = JSON.parse(localStorage.getItem(ORDER_KEY) ?? '[]')
    const ids = items.map((i) => i.id)
    localStorage.setItem(ORDER_KEY, JSON.stringify([...ids, ...saved.filter((s) => !ids.includes(s))]))
  } catch { /* storage unavailable: the order just won't persist */ }
}

export function Inspector({ analysis: a, freshness: fr }: Props) {
  const ids: WidgetId[] = []
  if (!a || !fr) ids.push('analysis')
  else ids.push('ground', 'classes', 'depth', 'health', 'source')

  const render = (id: WidgetId): ReactNode => {
    if (id === 'analysis') {
      return (
        <Section title="Analysis">
          <p className="dim">No analysis available. Start a camera source; results appear once Dev 1's Perception Port is publishing.</p>
        </Section>
      )
    }
    if (!a || !fr) return null
    const { meta, depthStats: d } = a
    switch (id) {
      case 'ground':
        return (
          <Section title="Ground map">
            <TopDownMap analysis={a} freshness={fr} />
            <div className="mini-legend">
              <span><i style={{ background: '#153a4d' }} />FREE</span>
              <span><i style={{ background: '#2c2c2c' }} />UNKNOWN / INFLATED</span>
              <span><i style={{ background: '#ff2a2a' }} />LETHAL</span>
            </div>
            <p className="dim">Flat-ground preview from the mask, 0.25 m cells. Not Nav2's costmap. Unknown is never free.</p>
          </Section>
        )
      case 'classes':
        return (
          <Section title="Classes">
            {CLASS_NAMES.map((n, i) => (
              <div className="bar" key={n}>
                <span>{i} {n}</span>
                <div><i style={{ width: `${a.classPct[i]}%`, background: `rgb(${CLASS_RGB[i].join(',')})` }} /></div>
                <b>{a.classPct[i].toFixed(1)}%</b>
              </div>
            ))}
          </Section>
        )
      case 'depth':
        return (
          <Section title="Depth">
            {d ? (
              <>
                <Row k="min" v={`${meta.kAssumed ? '~' : ''}${d.min.toFixed(1)} m`} />
                <Row k="median" v={`${meta.kAssumed ? '~' : ''}${d.median.toFixed(1)} m`} />
                <Row k="max" v={`${meta.kAssumed ? '~' : ''}${d.max.toFixed(1)} m`} />
                <div className="scale"><span>{meta.kAssumed ? '~' : ''}1.5 m</span><i className="depth" /><span>15 m+</span></div>
              </>
            ) : (
              <p className="dim">No depth channel: the perception node publishes depth only once the DA3 weights are exported.</p>
            )}
          </Section>
        )
      case 'health':
        return (
          <Section title="Health">
            <Row k="perception_degraded" v={a.degraded ? 'TRUE' : 'FALSE'} tone={a.degraded ? 'trip' : undefined} />
            <Row
              k={meta.streaming ? 'mask age' : 'captured'}
              v={meta.streaming ? `${fr.ageMs} / ${PERCEPTION_MAX_AGE_MS} ms` : new Date(meta.stamp).toLocaleTimeString()}
              tone={fr.stale ? 'trip' : undefined}
            />
            <Row k="path" v={a.path.length ? 'FOUND' : 'BLOCKED'} tone={a.path.length ? undefined : 'trip'} />
            <Row k="calibration" v={meta.kAssumed ? 'K ASSUMED' : 'K MEASURED'} tone={meta.kAssumed ? 'warn' : undefined} />
            {a.reasons.map((r) => <p className="reason" key={r}>{r}</p>)}
            {meta.kAssumed && <p className="dim">No CameraInfo: depth and the ground map are not metric-valid.</p>}
          </Section>
        )
      case 'source':
        return (
          <Section title="Source">
            <Row k="source" v={meta.source.toUpperCase()} />
            <Row k="frame_id" v={meta.frameId} />
            <Row k="stamp" v={new Date(meta.stamp).toLocaleTimeString()} />
            <Row k="size" v={`${meta.width} x ${meta.height}`} />
            <Row k="fx / fy" v={`${meta.K.fx.toFixed(0)} / ${meta.K.fy.toFixed(0)}`} />
            <Row k="cx / cy" v={`${meta.K.cx.toFixed(0)} / ${meta.K.cy.toFixed(0)}`} />
            <Row k="latency" v={`${a.latencyMs.toFixed(0)} ms`} />
          </Section>
        )
    }
    return null
  }

  return (
    <div className="sidegroup">
      <h4 className="sidegroup-title">Perception</h4>
      {/* remount when the set of widgets changes; the grid keeps its own order between changes */}
      <DraggableWidgetGrid
        key={ids.join(',')}
        items={ordered(ids)}
        onChange={remember}
        renderItem={(item) => render(item.id as WidgetId)}
        maxColumns={2}
        cellSize={200}
        gap={12}
        radius={4}
      />
    </div>
  )
}
