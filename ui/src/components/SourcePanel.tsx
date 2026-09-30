import { useRef, useState, type DragEvent } from 'react'
import type { RosOptions } from '../source/rosbridge'
import type { Layers, SourceKind } from '../types'
import { GoalPanel } from './GoalPanel'

interface Props {
  source: SourceKind
  onSource: (s: SourceKind) => void
  layers: Layers
  onLayers: (l: Layers) => void
  live: boolean
  onLive: (on: boolean) => void
  onPhoto: () => void
  onUpload: (f: File) => void
  rosCfg: RosOptions
  onRosCfg: (c: RosOptions) => void
  rosOn: boolean
  onRosOn: (on: boolean) => void
  rosConnected: boolean
  goalStatus: string
  onSendGoal: (fwd: number, left: number, relYawRad: number) => void
  onCancelGoal: () => void
  canSendGoal: boolean
  goalBlockedReason: string
  hasStart: boolean
  canSetStart: boolean
  onSetStart: () => void
  estop: boolean
  onEstop: (asserted: boolean) => void
}

const TABS: [SourceKind, string][] = [['upload', 'UPLOAD'], ['camera', 'CAMERA'], ['ros2', 'ROS 2']]
const LAYERS: [keyof Layers, string][] = [['image', 'Image'], ['mask', 'Class mask'], ['depth', 'Depth'], ['path', 'Path']]

export function SourcePanel(p: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

  const drop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    const f = e.dataTransfer.files[0]
    if (f?.type.startsWith('image/')) p.onUpload(f)
  }

  return (
    <aside className="panel source">
      <section className="section">
        <h3>Source</h3>
        <div className="tabs">
          {TABS.map(([k, label]) => (
            <button key={k} className={p.source === k ? 'on' : ''} onClick={() => p.onSource(k)}>{label}</button>
          ))}
        </div>

        {p.source === 'upload' && (
          <div
            className={`drop ${over ? 'over' : ''}`}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setOver(true) }}
            onDragLeave={() => setOver(false)}
            onDrop={drop}
          >
            <b>Drop a photo</b>
            <span>or click to browse</span>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => { const f = e.target.files?.[0]; if (f) p.onUpload(f); e.target.value = '' }}
            />
          </div>
        )}

        {p.source === 'camera' && (
          <div className="stack">
            <button className={`btn ${p.live ? '' : 'primary'}`} onClick={() => p.onLive(!p.live)}>
              {p.live ? 'STOP LIVE' : 'START LIVE DETECT'}
            </button>
            <button className="btn" onClick={p.onPhoto}>TAKE PHOTO</button>
          </div>
        )}

        {p.source === 'ros2' && (
          <div className="stack">
            <label>rosbridge URL
              <input value={p.rosCfg.url} disabled={p.rosOn} onChange={(e) => p.onRosCfg({ ...p.rosCfg, url: e.target.value })} />
            </label>
            <label>Image topic (CompressedImage)
              <input value={p.rosCfg.imageTopic} disabled={p.rosOn} onChange={(e) => p.onRosCfg({ ...p.rosCfg, imageTopic: e.target.value })} />
            </label>
            <label>CameraInfo topic
              <input value={p.rosCfg.infoTopic} disabled={p.rosOn} onChange={(e) => p.onRosCfg({ ...p.rosCfg, infoTopic: e.target.value })} />
            </label>
            <button className={`btn ${p.rosOn ? '' : 'primary'}`} onClick={() => p.onRosOn(!p.rosOn)}>
              {p.rosOn ? 'DISCONNECT' : 'CONNECT'}
            </button>
          </div>
        )}
      </section>

      {p.source === 'ros2' && (
        <GoalPanel
          connected={p.rosConnected} status={p.goalStatus}
          onSend={p.onSendGoal} onCancel={p.onCancelGoal}
          canSend={p.canSendGoal} blockedReason={p.goalBlockedReason}
          hasStart={p.hasStart} canSetStart={p.canSetStart} onSetStart={p.onSetStart}
          estop={p.estop} onEstop={p.onEstop}
        />
      )}

      <section className="section">
        <h3>Layers</h3>
        {LAYERS.map(([k, label]) => (
          <label className="toggle" key={k}>
            <input type="checkbox" checked={p.layers[k]} onChange={(e) => p.onLayers({ ...p.layers, [k]: e.target.checked })} />
            <i />
            {label}
          </label>
        ))}
      </section>
    </aside>
  )
}
