import type { Status } from '../types'

const LABEL: Record<Status, string> = { idle: 'IDLE', live: 'LIVE', still: 'STILL', error: 'ERROR' }

interface Props {
  status: Status
  fps: number
  note: string
  backendOnline: boolean
  models?: { seg: string; depth: string }
}

export function TopBar({ status, fps, note, backendOnline, models }: Props) {
  return (
    <header className="topbar">
      <div className="brand">WHOAMI<span>/ perception workbench</span></div>
      {backendOnline ? (
        <div className="engine real" title={models ? `${models.seg} + ${models.depth}` : 'local backend'}>REAL ANALYSIS</div>
      ) : (
        <div className="engine mock" title="Local backend (server/) is offline or unreachable — showing a placeholder mask and depth">MOCK ANALYSIS</div>
      )}
      {note && <div className={`note ${status === 'error' ? 'err' : ''}`}>{note}</div>}
      <div className={`pill ${status}`}>
        <i />
        {LABEL[status]}
        {status === 'live' && <em>{fps.toFixed(1)} fps</em>}
      </div>
    </header>
  )
}
