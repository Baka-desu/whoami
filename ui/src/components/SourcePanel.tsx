import { useRef, useState, type DragEvent } from 'react'
import type { CameraSource } from '../source/useCameraSource'
import type { SourceKind } from '../types'

const TABS: [SourceKind, string][] = [['ros2', 'ROBOT'], ['camera', 'BROWSER'], ['upload', 'PHOTO']]

// What feeds the camera view. The robot camera (rosbridge) is the default; the browser camera and photo uploads
// are for bench checks and have no perception backend.
export function SourcePanel({ cam }: { cam: CameraSource }) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

  const drop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    const f = e.dataTransfer.files[0]
    if (f?.type.startsWith('image/')) void cam.upload(f)
  }

  return (
    <section className="section">
      <h3>Camera source</h3>
      <div className="tabs">
        {TABS.map(([k, label]) => (
          <button key={k} className={cam.source === k ? 'on' : ''} onClick={() => cam.pickSource(k)}>{label}</button>
        ))}
      </div>

      {cam.source === 'ros2' && (
        <div className="stack">
          <label>rosbridge URL
            <input value={cam.rosCfg.url} disabled={cam.rosOn} onChange={(e) => cam.setRosCfg({ ...cam.rosCfg, url: e.target.value })} />
          </label>
          <label>Image topic (CompressedImage)
            <input value={cam.rosCfg.imageTopic} disabled={cam.rosOn} onChange={(e) => cam.setRosCfg({ ...cam.rosCfg, imageTopic: e.target.value })} />
          </label>
          <label>CameraInfo topic
            <input value={cam.rosCfg.infoTopic} disabled={cam.rosOn} onChange={(e) => cam.setRosCfg({ ...cam.rosCfg, infoTopic: e.target.value })} />
          </label>
          <button className={`btn ${cam.rosOn ? '' : 'primary'}`} onClick={() => cam.toggleRos(!cam.rosOn)}>
            {cam.rosOn ? 'DISCONNECT' : 'CONNECT'}
          </button>
          <p className="dim">{cam.rosConnected ? 'Reading the camera + Dev 1 Perception Port (read only).' : 'Not connected.'}</p>
        </div>
      )}

      {cam.source === 'camera' && (
        <div className="stack">
          <button className={`btn ${cam.live ? '' : 'primary'}`} onClick={() => cam.setLiveCamera(!cam.live)}>
            {cam.live ? 'STOP LIVE' : 'START LIVE'}
          </button>
          <button className="btn" onClick={() => void cam.takePhoto()}>TAKE PHOTO</button>
        </div>
      )}

      {cam.source === 'upload' && (
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
            onChange={(e) => { const f = e.target.files?.[0]; if (f) void cam.upload(f); e.target.value = '' }}
          />
        </div>
      )}
    </section>
  )
}
