import { Component, lazy, Suspense, useEffect, useState, type ReactNode } from 'react'
import { CameraView } from './components/CameraView'
import { CommandPanel } from './components/CommandPanel'
import { Inspector } from './components/Inspector'
import { SafetyBoard } from './components/SafetyBoard'
import { SourcePanel } from './components/SourcePanel'
import { StatusWidgets } from './components/StatusWidgets'
import { TopBar } from './components/TopBar'
import {
  parseView, slotOnError, slotOnProps, slotState, type MainView, type SlotFailure, type SlotState,
} from './map/mapToggles'
import { ApiError, api, isLive, subscribeTelemetry, type Mode, type Telemetry } from './source/api'
import { useCameraSource } from './source/useCameraSource'

const CLOCK_MS = 250 // re-check telemetry freshness at 4 Hz so a dead stream reads NO SIGNAL promptly

// three.js is large, so the map view is its own lazily loaded chunk: the camera view's first paint does not pay for it.
const MapView = lazy(() => import('./components/MapView').then((m) => ({ default: m.MapView })))

const VIEWS: MainView[] = ['camera', 'map']
const VIEW_KEY = 'ugv.console.view'

const savedView = (): MainView => {
  try {
    return parseView(localStorage.getItem(VIEW_KEY))
  } catch {
    return 'camera'
  }
}

interface SlotProps {
  resetKey: string // a new key (another view, another attempt) clears a failure
  fallback: (failure: SlotFailure) => ReactNode
  children: ReactNode
}

// Error boundary around the main view only. A view that fails - its code cannot be downloaded (a stale chunk after a
// redeploy, a network hiccup) or it throws while rendering or in an effect (say, a malformed frame) - shows its
// failure in the view area; without this React would unmount the whole console, the e-stop and safety board with it.
// The state logic is in mapToggles.ts (slotState / slotOnError / slotOnProps), where it is tested.
class ViewBoundary extends Component<SlotProps, SlotState> {
  state: SlotState = slotState(this.props.resetKey)

  static getDerivedStateFromError(error: unknown) {
    return slotOnError(error)
  }

  static getDerivedStateFromProps(props: SlotProps, state: SlotState) {
    return slotOnProps(state, props.resetKey)
  }

  render() {
    return this.state.failed ? this.props.fallback(this.state.failed) : this.props.children
  }
}

type Message = { text: string; reasons?: string[]; error: boolean } | null

const fromError = (e: unknown): Message =>
  e instanceof ApiError
    ? { text: `${e.problem.title}${e.problem.detail ? `: ${e.problem.detail}` : ''}`, reasons: e.problem.reasons, error: true }
    : { text: String(e), error: true }

// Operator console. The main area shows one of two views, picked in the top bar: the camera view (Dev 1's mask /
// depth / path over the live image) or the 3D map view; both are display only, and only the shown one is mounted.
// Left sidebar: the operator's commands (e-stop §3.1, mapping|localize §10, map-frame goal §11) through Dev 5's
// gateway (/api/v1), and the camera source. Right sidebar: §12 health table, robot status, perception details.
export default function App() {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [connected, setConnected] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const [estopBusy, setEstopBusy] = useState(false)
  const [message, setMessage] = useState<Message>(null)
  const [view, setView] = useState<MainView>(savedView)
  const [viewAttempt, setViewAttempt] = useState(0) // bumped by "try again" after a failed view
  const cam = useCameraSource()

  const pickView = (v: MainView) => {
    setView(v)
    try { localStorage.setItem(VIEW_KEY, v) } catch { /* not persisted */ }
  }

  const viewFailed = (f: SlotFailure) => (
    <section className="viewfail" role="alert">
      <b>{view} view {f.load ? 'failed to load' : 'stopped'}</b>
      <span>{f.detail}</span>
      <span>The rest of the console, the e-stop included, keeps working.</span>
      <div className="viewfail-actions">
        {view === 'map' && (
          <button type="button" className="btn primary" onClick={() => pickView('camera')}>back to camera view</button>
        )}
        {f.load ? (
          <button type="button" className="btn" onClick={() => window.location.reload()}>reload page</button>
        ) : (
          <button type="button" className="btn" onClick={() => setViewAttempt((a) => a + 1)}>try again</button>
        )}
      </div>
    </section>
  )

  useEffect(() => subscribeTelemetry(setTelemetry, setConnected), [])
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), CLOCK_MS)
    return () => window.clearInterval(t)
  }, [])

  const live = isLive(telemetry, now)
  const safety = live ? telemetry.safety : undefined
  const gateReasons = safety ? safety.watches.filter((w) => !w.ok).map((w) => `${w.name}: ${w.reason}`) : []

  const run = async (fn: () => Promise<string>) => {
    try {
      setMessage({ text: await fn(), error: false })
    } catch (e) {
      setMessage(fromError(e))
    }
  }

  const setEstop = async (asserted: boolean) => {
    setEstopBusy(true)
    await run(async () => {
      const r = await api.setEstop(asserted)
      return r.assertedByGateway ? 'E-stop asserted.' : 'E-stop released by this console.'
    })
    setEstopBusy(false)
  }

  const setMode = (mode: Mode) => run(async () => `Localization mode set to ${await api.setMode(mode)}.`)

  const sendGoal = (x: number, y: number, yaw: number) =>
    run(async () => {
      const g = await api.sendGoal(x, y, yaw)
      return `Goal ${g.id.slice(0, 8)} sent: ${g.state}.`
    })

  const cancelGoal = (id: string) => run(async () => `Goal ${id.slice(0, 8)}: ${(await api.cancelGoal(id)).state}.`)

  return (
    <div className="app">
      <TopBar connected={connected} live={live} safetyOk={safety?.ok ?? null} />
      <nav className="viewswitch" aria-label="Main view">
        {VIEWS.map((v) => (
          <button key={v} type="button" className={view === v ? 'on' : ''} aria-pressed={view === v} onClick={() => pickView(v)}>
            {v}
          </button>
        ))}
      </nav>
      <aside className="panel source">
        <CommandPanel
          live={live}
          estopAsserted={safety?.eStop.asserted ?? false}
          estopBusy={estopBusy}
          onEstop={setEstop}
          mode={live ? telemetry.localization?.requestedMode ?? null : null}
          onMode={setMode}
          gateReasons={gateReasons}
          activeGoal={live ? telemetry.navigation?.activeGoal ?? null : null}
          onSendGoal={sendGoal}
          onCancelGoal={cancelGoal}
          message={message}
        />
        <SourcePanel cam={cam} />
      </aside>
      <ViewBoundary resetKey={`${view}:${viewAttempt}`} fallback={viewFailed}>
        {view === 'camera' ? (
          <CameraView cam={cam} />
        ) : (
          <Suspense fallback={<section className="mapview" aria-busy="true" />}>
            <MapView telemetry={telemetry} live={live} />
          </Suspense>
        )}
      </ViewBoundary>
      <aside className="panel inspector">
        <SafetyBoard safety={safety} live={live} />
        <p className="inspector-hint">drag widgets to rearrange · alt + arrows on keyboard</p>
        <StatusWidgets
          live={live}
          command={telemetry?.command}
          navigation={telemetry?.navigation}
          localization={telemetry?.localization}
          eStop={safety?.eStop}
          map={telemetry?.map}
        />
        <Inspector analysis={cam.analysis} freshness={cam.freshness} />
      </aside>
    </div>
  )
}
