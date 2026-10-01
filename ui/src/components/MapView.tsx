import { useCallback, useEffect, useRef, useState } from 'react'
import type { DepthFrame } from '../map/codec'
import { depthToRgba, type ElevationColorMode } from '../map/geometry'
import { MapScene } from '../map/scene'
import { enabledLayers, parseToggles, type MapToggles } from '../map/scene-math'
import { useMapData } from '../map/useMapData'
import { LAYERS, type MapStatus, type Telemetry } from '../source/api'

// The map view, as in the owner's reference picture: the 3D map (accumulated cloud, live depth scan coloured by
// height, elevation, trajectory, the cost grid's halo around obstacles, the robot) with the depth image and the
// camera image stacked on its left edge. Display only: everything comes from the gateway's map endpoints (GETs) and
// the telemetry stream's pose; nothing here commands anything.

const TOGGLES_KEY = 'ugv.console.map.layers'
const CAMERA_PANEL_MAX_W = 480 // the camera panel is a thumbnail; no need to keep a full-size copy

type ToggleKey = Exclude<keyof MapToggles, 'mode'>
const LAYER_BUTTONS: { key: ToggleKey; label: string; title: string }[] = [
  { key: 'cloud', label: 'map cloud', title: 'Accumulated 3D map, in camera colours' },
  { key: 'live', label: 'live cloud', title: 'The current depth scan, coloured by height' },
  { key: 'trajectory', label: 'trajectory', title: 'Path the robot has travelled' },
  { key: 'elevation', label: 'elevation', title: 'Elevation map of the ground' },
  { key: 'grid', label: 'cost grid', title: 'Navigation costmap: obstacles and their inflation halo' },
  { key: 'images', label: 'images', title: 'Depth and camera panels' },
]
const MODES: { value: ElevationColorMode; label: string }[] = [
  { value: 'height', label: 'height' },
  { value: 'confidence', label: 'confidence' },
  { value: 'obstacle', label: 'obstacle' },
]

const loadToggles = (): MapToggles => {
  try {
    return parseToggles(localStorage.getItem(TOGGLES_KEY))
  } catch {
    return parseToggles(null)
  }
}

const remember = (t: MapToggles) => {
  try { localStorage.setItem(TOGGLES_KEY, JSON.stringify(t)) } catch { /* not persisted */ }
}

// A frame from an earlier gateway process (its epoch is not the current status's) is not shown: after a restart the
// hook keeps old frames until each layer is fetched again, and a layer still at seq 0 never is.
function ofEpoch<T extends { epoch: number }>(frame: T | null, status: MapStatus | null): T | null {
  return frame && status && frame.epoch === status.epoch ? frame : null
}

// Polling pauses while the tab is hidden: nobody is looking, and the gateway drops its heavy subscriptions.
function usePageVisible(): boolean {
  const [visible, setVisible] = useState(() => !document.hidden)
  useEffect(() => {
    const onChange = () => setVisible(!document.hidden)
    document.addEventListener('visibilitychange', onChange)
    return () => document.removeEventListener('visibilitychange', onChange)
  }, [])
  return visible
}

interface Props {
  telemetry: Telemetry | null
  live: boolean // telemetry stream fresh
}

export function MapView({ telemetry, live }: Props) {
  const [toggles, setToggles] = useState<MapToggles>(loadToggles)
  const pageVisible = usePageVisible()
  const data = useMapData(pageVisible, enabledLayers(toggles)) // a toggled-off layer is not fetched
  const [scene, setScene] = useState<MapScene | null>(null)
  const [glFailed, setGlFailed] = useState(false)

  // The scene lives exactly as long as its host element: one WebGL context per mount, created when React attaches the
  // element and disposed (context released) when it detaches. StrictMode attaches, detaches and attaches again in
  // development; the second attach gets a fresh scene after the first one is disposed. Stable (no deps), so a
  // re-render never re-attaches.
  const attachScene = useCallback((host: HTMLDivElement | null) => {
    if (!host) return
    let s: MapScene
    try {
      s = new MapScene(host)
    } catch {
      setGlFailed(true) // no WebGL: say so in the view instead of throwing
      return
    }
    s.resize(host.clientWidth, host.clientHeight)
    const ro = new ResizeObserver(() => s.resize(host.clientWidth, host.clientHeight))
    ro.observe(host)
    setScene(s)
    return () => {
      ro.disconnect()
      s.dispose()
      setScene(null)
    }
  }, [])

  const status = data.status
  const noMap = !status || LAYERS.every((l) => status.seq[l] === 0)
  const pose = telemetry?.pose ?? null
  const cloud = ofEpoch(data.cloud, status)
  const liveCloud = ofEpoch(data.live, status)
  const elevation = ofEpoch(data.elevation, status)
  const trajectory = ofEpoch(data.trajectory, status)
  const grid = ofEpoch(data.grid, status)
  const depth = ofEpoch(data.depth, status)
  const camera = status && status.seq.camera > 0 ? data.camera : null // a bitmap carries no epoch

  // The pose goes first: the first layer to arrive frames the camera on it within the same commit.
  useEffect(() => { scene?.setPose(pose) }, [scene, pose])
  const { cloud: showCloud, live: showLive, trajectory: showTrajectory, elevation: showElevation, grid: showGrid, mode } = toggles
  useEffect(() => {
    scene?.setLayers({ cloud: showCloud, live: showLive, trajectory: showTrajectory, elevation: showElevation, grid: showGrid })
  }, [scene, showCloud, showLive, showTrajectory, showElevation, showGrid])
  useEffect(() => { scene?.setCloud(cloud) }, [scene, cloud])
  useEffect(() => { scene?.setLive(liveCloud) }, [scene, liveCloud])
  useEffect(() => { scene?.setElevation(elevation, mode) }, [scene, elevation, mode])
  useEffect(() => { scene?.setTrajectory(trajectory) }, [scene, trajectory])
  useEffect(() => { scene?.setGrid(grid) }, [scene, grid])

  const update = (patch: Partial<MapToggles>) => {
    const next = { ...toggles, ...patch }
    setToggles(next)
    remember(next)
  }

  const staleReason = data.stale ? 'map not updating' : !live ? 'telemetry lost' : null
  const ok = !noMap && staleReason === null
  const slamMode = typeof status?.stats.mode === 'string' ? status.stats.mode : null
  const depthPanel = toggles.images ? depth : null // a panel shows only with data and with the images toggle on
  const cameraPanel = toggles.images ? camera : null

  return (
    <section className="mapview">
      <header className="livefeed-bar">
        <span className={`dot ${ok ? 'ok' : 'bad'}`} />
        <span>
          3d map
          {slamMode && ` · ${slamMode}`}
          {cloud && ` · ${cloud.count.toLocaleString('en-US')} pts`}
        </span>
        <span className="livefeed-layers">
          {LAYER_BUTTONS.map(({ key, label, title }) => (
            <button key={key} type="button" title={title} className={toggles[key] ? 'on' : ''} aria-pressed={toggles[key]}
              onClick={() => update({ [key]: !toggles[key] })}>
              {label}
            </button>
          ))}
          <select aria-label="Elevation colour" title="Elevation colour" value={toggles.mode} disabled={!toggles.elevation}
            onChange={(e) => update({ mode: e.target.value as ElevationColorMode })}>
            {MODES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
          </select>
          <button type="button" title="Back behind the robot" disabled={!scene} onClick={() => scene?.resetView()}>
            reset view
          </button>
        </span>
      </header>
      <div className="mapview-stage">
        <div ref={attachScene} className="mapview-gl" />
        {glFailed ? (
          <div className="nosignal">
            <b>NO 3D VIEW</b>
            <span>WebGL could not start in this browser</span>
          </div>
        ) : noMap ? (
          <div className="nosignal">
            <b>NO MAP YET</b>
            <span>{data.stale ? 'waiting for the gateway' : 'no map layer has data yet'}</span>
          </div>
        ) : null}
        {!noMap && staleReason && <div className="banner stale">STALE · {staleReason}</div>}
        {(depthPanel || cameraPanel) && (
          <div className="mapview-insets">
            {depthPanel && <DepthPanel frame={depthPanel} />}
            {cameraPanel && <CameraPanel frame={cameraPanel} />}
          </div>
        )}
      </div>
    </section>
  )
}

// Depth, grey: near is bright, holes show the panel behind them.
function DepthPanel({ frame }: { frame: DepthFrame }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c || frame.width === 0 || frame.height === 0) return
    if (c.width !== frame.width) c.width = frame.width
    if (c.height !== frame.height) c.height = frame.height
    // depthToRgba allocates a plain ArrayBuffer, which is what ImageData wants; the cast only says so.
    const rgba = depthToRgba(frame) as Uint8ClampedArray<ArrayBuffer>
    c.getContext('2d')?.putImageData(new ImageData(rgba, frame.width, frame.height), 0, 0)
  }, [frame])
  return (
    <figure className="mapview-inset" title="Depth image: near is bright">
      <figcaption>depth</figcaption>
      <canvas ref={ref} />
    </figure>
  )
}

function CameraPanel({ frame }: { frame: ImageBitmap }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c || frame.width === 0 || frame.height === 0) return // a released bitmap reads 0 x 0
    const w = Math.min(frame.width, CAMERA_PANEL_MAX_W)
    const h = Math.round((w * frame.height) / frame.width)
    if (c.width !== w) c.width = w
    if (c.height !== h) c.height = h
    try {
      c.getContext('2d')?.drawImage(frame, 0, 0, w, h)
    } catch {
      // released between render and effect: the next frame draws
    }
  }, [frame])
  return (
    <figure className="mapview-inset">
      <figcaption>camera</figcaption>
      <canvas ref={ref} />
    </figure>
  )
}
