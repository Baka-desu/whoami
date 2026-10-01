import type { ReactNode } from 'react'
import type { BaseCommand, EStop, Localization, MapStatus, Navigation } from '../source/api'
import { mapRows, placeholderCalibration } from './mapStats'
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

const flag = (v: boolean | null | undefined, yes: string, no: string) => (v === null || v === undefined ? 'NO SIGNAL' : v ? yes : no)
const ageMs = (s: number | null) => (s === null ? '—' : `${(s * 1000).toFixed(0)} ms`)

interface Props {
  live: boolean
  command?: BaseCommand
  navigation?: Navigation
  localization?: Localization
  eStop?: EStop
  map?: MapStatus
}

type WidgetId = 'command' | 'navigation' | 'localization' | 'estop' | 'map'
const SIZE: Record<WidgetId, WidgetSize> = { command: 'wide', navigation: 'lg', localization: 'wide', estop: 'wide', map: 'lg' }
const LABEL: Record<WidgetId, string> = { command: 'Final /cmd_vel', navigation: 'Navigation', localization: 'Localization', estop: 'E-stop', map: 'Map' }
const IDS: WidgetId[] = ['command', 'navigation', 'localization', 'estop', 'map']
const ORDER_KEY = 'ugv.console.widgets.order'

function ordered(): WidgetItem[] {
  let saved: string[] = []
  try { saved = JSON.parse(localStorage.getItem(ORDER_KEY) ?? '[]') } catch { /* ignore */ }
  const rank = (id: string) => { const i = saved.indexOf(id); return i < 0 ? 999 : i }
  return [...IDS].sort((a, b) => rank(a) - rank(b)).map((id) => ({ id, size: SIZE[id], label: LABEL[id] }))
}

function remember(items: WidgetItem[]) {
  try { localStorage.setItem(ORDER_KEY, JSON.stringify(items.map((i) => i.id))) } catch { /* not persisted */ }
}

// Read-only views of the gateway resources. A stalled stream shows NO SIGNAL, never the last value.
export function StatusWidgets({ live, command, navigation: nav, localization: loc, eStop, map }: Props) {
  const render = (id: WidgetId): ReactNode => {
    if (!live) return <Section title={LABEL[id]}><p className="dim">NO SIGNAL</p></Section>
    switch (id) {
      case 'command':
        return (
          <Section title="Final /cmd_vel">
            {command?.available && command.linear && command.angular ? (
              <>
                <Row k="linear x" v={`${command.linear.x.toFixed(2)} m/s`} />
                <Row k="angular z" v={`${command.angular.z.toFixed(2)} rad/s`} />
                <Row k="age" v={ageMs(command.ageS)} tone={(command.ageS ?? 0) > 0.5 ? 'warn' : undefined} />
              </>
            ) : (
              <Row k="/cmd_vel" v="NOT PUBLISHED" tone="warn" />
            )}
            <p className="dim">What the safety authority let through to the base (§3.1). Read only.</p>
          </Section>
        )
      case 'navigation': {
        const g = nav?.activeGoal
        return (
          <Section title="Navigation">
            <Row k="nav2 heartbeat" v={flag(nav?.heartbeat, 'ACTIVE', 'DOWN')} tone={nav?.heartbeat ? undefined : 'trip'} />
            {nav?.status && nav.status !== 'ok' && <p className="reason">{nav.status}</p>}
            <Row k="/navigate_to_pose" v={nav?.actionServerReady ? 'READY' : 'UNAVAILABLE'} tone={nav?.actionServerReady ? undefined : 'warn'} />
            {g ? (
              <>
                <Row k="goal" v={`${g.x.toFixed(1)}, ${g.y.toFixed(1)} (${g.frameId})`} />
                <Row k="state" v={g.state.toUpperCase()} />
                {g.distanceRemaining !== null && <Row k="remaining" v={`${g.distanceRemaining.toFixed(1)} m`} />}
                {g.recoveries !== null && g.recoveries > 0 && <Row k="recoveries" v={g.recoveries} tone="warn" />}
                {g.errorMessage && <p className="reason">{g.errorMessage}</p>}
              </>
            ) : (
              <Row k="goal" v="none" />
            )}
          </Section>
        )
      }
      case 'localization':
        return (
          <Section title="Localization">
            <Row k="pose_valid" v={flag(loc?.poseValid, 'TRUE', 'FALSE')} tone={loc?.poseValid ? undefined : 'trip'} />
            {loc?.status && loc.status !== 'valid' && <p className="reason">{loc.status.split(',').join(' · ')}</p>}
            <Row k="mode (set here)" v={loc?.requestedMode?.toUpperCase() ?? 'NOT SET'} />
          </Section>
        )
      case 'estop':
        return (
          <Section title="E-stop">
            <Row k="state" v={eStop?.asserted ? 'ASSERTED' : 'released'} tone={eStop?.asserted ? 'trip' : undefined} />
            <Row k="by this console" v={eStop?.assertedByGateway ? 'YES' : 'no'} />
            <Row k="last on /ugv/e_stop" v={eStop?.lastSeen === null || eStop?.lastSeen === undefined ? 'never' : String(eStop.lastSeen)} />
          </Section>
        )
      case 'map':
        return (
          <Section title="Map">
            {mapRows(map?.stats).map((r) => <Row key={r.k} k={r.k} v={r.v} />)}
            {placeholderCalibration(map?.stats) && <p className="reason">placeholder calibration</p>}
          </Section>
        )
    }
  }

  return (
    <div className="sidegroup">
      <h4 className="sidegroup-title">Robot</h4>
      <DraggableWidgetGrid
        items={ordered()}
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
