import { useCallback, useEffect, useRef, useState } from 'react'
import type { DepthFrame } from '../map/codec'
import { depthToRgba, type ElevationColorMode } from '../map/geometry'
import { enabledLayers, parseToggles, toggled, type MapToggles, type ToggleKey } from '../map/mapToggles'
import { MapScene } from '../map/scene'
import { useMapData } from '../map/useMapData'
import { LAYERS, type MapStatus, type Telemetry } from '../source/api'

// The map view, as in the owner's reference picture: the 3D map (accumulated cloud, live depth scan coloured by
// height, elevation, trajectory, the cost grid's halo around obstacles, the robot) with the depth image and the
// camera image stacked on its left edge. Display only: everything comes from the gateway's map endpoints (GETs) and
// the telemetry stream's pose; nothing here commands anything.

const TOGGLES_KEY = 'ugv.console.map.layers'
const CAMERA_PANEL_MAX_W = 480 // the camera panel is a thumbnail; no need to keep a full-size copy

// Short labels keep the bar on one row from a 1280 px window up (the view area is then about 580 px wide); the full
// name is the accessible name and the tooltip.
const LAYER_BUTTONS: { key: ToggleKey; label: string; name: string; title: string }[] = [
  { key: 'cloud', label: 'cloud', name: 'Map cloud', title: 'Map cloud: the accumulated 3D map, in camera colours' },
  { key: 'live', label: 'live', name: 'Live cloud', title: 'Live cloud: the current depth scan, coloured by height' },
  { key: 'trajectory', label: 'path', name: 'Trajectory path', title: 'Trajectory: the path the robot has travelled' },
  { key: 'elevation', label: 'elev', name: 'Elevation', title: 'Elevation map of the ground' },
  { key: 'grid', label: 'cost', name: 'Cost grid', title: 'Cost grid: obstacles and their inflation halo' },
  { key: 'images', label: 'img', name: 'Image panels', title: 'Image panels: depth and camera' },
]
const MODES: { value: ElevationColorMode; label: string }[] = [
  { value: 'height', label: 'height' },
  { value: 'confidence', label: 'conf' },
  { value: 'obstacle', label: 'obst' },
]
const count = (n: number) => n.toLocaleString('en-US')

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
  // Saved only once the operator changed something, so a later change of the defaults still reaches anyone who never
  // touched a toggle.
  const touched = useRef(false)
  useEffect(() => {
    if (touched.current) remember(toggles)
  }, [toggles])
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

  // The robot is hidden while telemetry is not live: a pose from a dead stream is not where the robot is. The pose
  // goes first: the first layer to arrive frames the camera on it within the same commit.
  const shownPose = live ? pose : null
  useEffect(() => { scene?.setPose(shownPose) }, [scene, shownPose])
  const { cloud: showCloud, live: showLive, trajectory: showTrajectory, elevation: showElevation, grid: showGrid, mode } = toggles
  useEffect(() => {
    scene?.setLayers({ cloud: showCloud, live: showLive, trajectory: showTrajectory, elevation: showElevation, grid: showGrid })
  }, [scene, showCloud, showLive, showTrajectory, showElevation, showGrid])
  useEffect(() => { scene?.setCloud(cloud) }, [scene, cloud])
  useEffect(() => { scene?.setLive(liveCloud) }, [scene, liveCloud])
  useEffect(() => { scene?.setElevation(elevation, mode) }, [scene, elevation, mode])
  useEffect(() => { scene?.setTrajectory(trajectory) }, [scene, trajectory])
  useEffect(() => { scene?.setGrid(grid) }, [scene, grid])

  // Functional updates: two clicks inside one render both count.
  const flip = (key: ToggleKey) => {
    touched.current = true
    setToggles((t) => toggled(t, key))
  }
  const pickMode = (m: ElevationColorMode) => {
    touched.current = true
    setToggles((t) => ({ ...t, mode: m }))
  }

  const staleReason = data.stale ? 'map not updating' : !live ? 'telemetry lost' : null
  const ok = !noMap && staleReason === null
  const slamMode = typeof status?.stats.mode === 'string' ? status.stats.mode : null
  const depthPanel = toggles.images ? depth : null // a panel shows only with data and with the images toggle on
  const cameraPanel = toggles.images ? camera : null
  // Counts sit in the stage, not the bar, so the bar never cuts a number short.
  const info = [
    showCloud && cloud && `${count(cloud.count)} map pts`,
    showLive && liveCloud && `${count(liveCloud.count)} live pts`,
    showTrajectory && trajectory && `${trajectory.lengthM.toFixed(1)} m path`,
  ].filter(Boolean).join(' · ')

  return (
    <section className="mapview">
      <header className="livefeed-bar">
        <span className={`dot ${ok ? 'ok' : 'bad'}`} />
        <span>
          3d map
          {slamMode && ` · ${slamMode}`}
        </span>
        <span className="livefeed-layers">
          {LAYER_BUTTONS.map(({ key, label, name, title }) => (
            <button key={key} type="button" title={title} aria-label={name} className={toggles[key] ? 'on' : ''}
              aria-pressed={toggles[key]} onClick={() => flip(key)}>
              {label}
            </button>
          ))}
          <select aria-label="Elevation colour" title="Elevation colour: height, confidence or obstacle" value={toggles.mode}
            disabled={!toggles.elevation} onChange={(e) => pickMode(e.target.value as ElevationColorMode)}>
            {MODES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
          </select>
          <button type="button" aria-label="Reset view" title="Reset view: back behind the robot" disabled={!scene}
            onClick={() => scene?.resetView()}>
            reset
          </button>
        </span>
      </header>
      <div className="mapview-stage">
        <div ref={attachScene} className="mapview-gl" role="img"
          aria-label="3D map around the robot: drag to orbit, right-drag to pan, scroll to zoom" />
        {!noMap && info && <div className="mapview-info">{info}</div>}
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
      <canvas ref={ref} role="img" aria-label="Depth image from the camera, near is bright" />
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
      <canvas ref={ref} role="img" aria-label="Latest camera image" />
    </figure>
  )
}
