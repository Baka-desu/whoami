interface Props {
  connected: boolean // SSE connection open
  live: boolean // telemetry fresh
  safetyOk: boolean | null
}

export function TopBar({ connected, live, safetyOk }: Props) {
  const status = !connected ? 'error' : live ? 'live' : 'idle'
  const label = !connected ? 'GATEWAY OFFLINE' : live ? 'LIVE' : 'NO TELEMETRY'
  return (
    <header className="topbar">
      <div className="brand">UGV<span>/ operator console</span></div>
      {live && safetyOk === false && <div className="analyzer-off" title="A §12 watch has tripped">HOLD</div>}
      <a className="tb-link" href="#features">FEATURES ↓</a>
      <div className="note">/api/v1</div>
      <div className={`pill ${status}`}>
        <i />
        {label}
      </div>
    </header>
  )
}
