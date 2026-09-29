import type { Status } from '../types'

const LABEL: Record<Status, string> = { idle: 'IDLE', live: 'LIVE', still: 'STILL', error: 'ERROR' }

interface Props {
  status: Status
  fps: number
  note: string
}

export function TopBar({ status, fps, note }: Props) {
  return (
    <header className="topbar">
      <div className="brand">WHOAMI<span>/ perception workbench</span></div>
      <div className="mock" title="Segmentation and depth are placeholders until the real models are wired in">MOCK ANALYSIS</div>
      {note && <div className={`note ${status === 'error' ? 'err' : ''}`}>{note}</div>}
      <div className={`pill ${status}`}>
        <i />
        {LABEL[status]}
        {status === 'live' && <em>{fps.toFixed(1)} fps</em>}
      </div>
    </header>
  )
}
