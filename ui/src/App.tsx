import { useEffect, useState } from 'react'
import { CommandPanel } from './components/CommandPanel'
import { LiveFeed } from './components/LiveFeed'
import { SafetyBoard } from './components/SafetyBoard'
import { StatusWidgets } from './components/StatusWidgets'
import { TopBar } from './components/TopBar'
import { ApiError, api, isLive, subscribeTelemetry, type Mode, type Telemetry } from './source/api'

const CLOCK_MS = 250 // re-check telemetry freshness at 4 Hz so a dead stream reads NO SIGNAL promptly

type Message = { text: string; reasons?: string[]; error: boolean } | null

const fromError = (e: unknown): Message =>
  e instanceof ApiError
    ? { text: `${e.problem.title}${e.problem.detail ? `: ${e.problem.detail}` : ''}`, reasons: e.problem.reasons, error: true }
    : { text: String(e), error: true }

// Operator console: the five operator items architecture.md defines (e-stop §3.1, §12 health table, final
// /cmd_vel, mapping|localize §10, map-frame goal §11), all through Dev 5's gateway (/api/v1). The live camera
// with Dev 1's mask / depth / path overlay is display-only (components/LiveFeed).
export default function App() {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [connected, setConnected] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const [estopBusy, setEstopBusy] = useState(false)
  const [message, setMessage] = useState<Message>(null)

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
      <div className="center">
        <LiveFeed />
        <SafetyBoard safety={safety} live={live} />
      </div>
      <StatusWidgets
        live={live}
        command={telemetry?.command}
        navigation={telemetry?.navigation}
        localization={telemetry?.localization}
        eStop={safety?.eStop}
      />
    </div>
  )
}
