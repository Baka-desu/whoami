import type { ReactNode } from 'react'
import type { Freshness } from '../analysis/freshness'
import { PERCEPTION_MAX_AGE_MS, CLASS_NAMES, CLASS_RGB, type Analysis } from '../types'
import { TopDownMap } from './TopDownMap'

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="section">
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

export function Inspector({ analysis: a, freshness: fr }: { analysis: Analysis | null; freshness: Freshness | null }) {
  if (!a || !fr) {
    return (
      <aside className="panel inspector">
        <Section title="Analysis">
          <p className="dim">Waiting for a frame.</p>
        </Section>
      </aside>
    )
  }
  const { meta, depthStats: d } = a

  return (
    <aside className="panel inspector">
      <Section title="Ground map">
        <TopDownMap analysis={a} freshness={fr} />
        <div className="mini-legend">
          <span><i style={{ background: '#153a4d' }} />FREE</span>
          <span><i style={{ background: '#2c2c2c' }} />UNKNOWN / INFLATED</span>
          <span><i style={{ background: '#ff2a2a' }} />LETHAL</span>
        </div>
        <p className="dim">Costmap, 0.25 m cells. Unknown is never free.</p>
      </Section>

      <Section title="Classes">
        {CLASS_NAMES.map((n, i) => (
          <div className="bar" key={n}>
            <span>{i} {n}</span>
            <div><i style={{ width: `${a.classPct[i]}%`, background: `rgb(${CLASS_RGB[i].join(',')})` }} /></div>
            <b>{a.classPct[i].toFixed(1)}%</b>
          </div>
        ))}
      </Section>

      <Section title="Depth">
        <Row k="min" v={`${meta.kAssumed ? '~' : ''}${d.min.toFixed(1)} m`} />
        <Row k="median" v={`${meta.kAssumed ? '~' : ''}${d.median.toFixed(1)} m`} />
        <Row k="max" v={`${meta.kAssumed ? '~' : ''}${d.max.toFixed(1)} m`} />
        <div className="scale"><span>{meta.kAssumed ? '~' : ''}1.5 m</span><i className="depth" /><span>15 m+</span></div>
      </Section>

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

      <Section title="Source">
        <Row k="source" v={meta.source.toUpperCase()} />
        <Row k="frame_id" v={meta.frameId} />
        <Row k="stamp" v={new Date(meta.stamp).toLocaleTimeString()} />
        <Row k="size" v={`${meta.width} x ${meta.height}`} />
        <Row k="fx / fy" v={`${meta.K.fx.toFixed(0)} / ${meta.K.fy.toFixed(0)}`} />
        <Row k="cx / cy" v={`${meta.K.cx.toFixed(0)} / ${meta.K.cy.toFixed(0)}`} />
        <Row k="latency" v={`${a.latencyMs.toFixed(0)} ms`} />
      </Section>
    </aside>
  )
}
