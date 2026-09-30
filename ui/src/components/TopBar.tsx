import type { Status } from '../types'

const LABEL: Record<Status, string> = { idle: 'IDLE', live: 'LIVE', still: 'STILL', error: 'ERROR' }

interface Props {
  status: Status
  fps: number
  note: string
  analyzerOnline: boolean
}

export function TopBar({ status, fps, note, analyzerOnline }: Props) {
  return (
    <header className="topbar">
      <div className="brand">WHOAMI<span>/ perception workbench</span></div>
      {!analyzerOnline && (
        <div className="analyzer-off" title="No perception backend is connected, so frames are shown without a mask, depth or path">
          NO ANALYZER
        </div>
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
