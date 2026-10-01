import { useCallback, useEffect, useRef, useState } from 'react'
import { useFreshness } from '../analysis/freshness'
import { RosPerception } from '../analysis/ros-analyzer'
import { connectRos, type CameraCalibration, type RosOptions } from '../source/rosbridge'
import type { Analysis, FrameMeta, Intrinsics, Layers } from '../types'
import { Viewport } from './Viewport'

// Live camera with Dev 1's Perception Port drawn over it (mask, metric depth, traversable path), read-only over
// rosbridge. Dev 5's driver publishes a JPEG stream for web UIs on /image_raw/compressed with /camera_info.
// Display only: nothing here can command motion; goals and e-stop stay on Dev 5's gateway.
const ROS: RosOptions = {
  url: `ws://${window.location.hostname || 'localhost'}:9090`,
  imageTopic: '/image_raw/compressed',
  infoTopic: '/camera_info',
}

const perception = new RosPerception()

function scaledK(calib: CameraCalibration | null, frameId: string, w: number, h: number): Intrinsics | null {
  if (!calib) return null
  const norm = (s: string) => s.replace(/^\//, '')
  if (norm(calib.frameId) && norm(calib.frameId) !== norm(frameId)) return null
  if (!(calib.width > 0 && calib.height > 0)) return calib.K
  const sx = w / calib.width
  const sy = h / calib.height
  return { fx: calib.K.fx * sx, fy: calib.K.fy * sy, cx: calib.K.cx * sx, cy: calib.K.cy * sy }
}

export function LiveFeed() {
  const [frame, setFrame] = useState<ImageBitmap | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [layers, setLayers] = useState<Layers>({ image: true, mask: true, depth: false, path: true })
  const [link, setLink] = useState<{ ok: boolean; text: string }>({ ok: false, text: 'connecting…' })
  const [fps, setFps] = useState(0)
  const calib = useRef<CameraCalibration | null>(null)
  const lastAt = useRef(0)
  const seq = useRef(0)
  const freshness = useFreshness(analysis)

  const ingest = useCallback(async (bmp: ImageBitmap, stamp: number, frameId: string) => {
    const K = scaledK(calib.current, frameId, bmp.width, bmp.height)
    if (!K) {
      bmp.close()
      return // no calibration for this frame yet: never draw geometry with an assumed K
    }
    const meta: FrameMeta = {
      source: 'ros2', frameId, stamp, receivedAt: Date.now(), width: bmp.width, height: bmp.height,
      K, kAssumed: false, streaming: true,
    }
    const now = performance.now()
    const dt = now - lastAt.current
    lastAt.current = now
    setFps((f) => (dt > 0 && dt < 2000 ? f * 0.7 + (1000 / dt) * 0.3 : 0))
    setFrame(bmp)
    const mine = ++seq.current
    let result: Analysis | null = null
    try {
      result = await perception.analyze(bmp, meta)
    } catch {
      result = null
    }
    if (mine === seq.current) setAnalysis(result)
  }, [])

  useEffect(() => () => frame?.close(), [frame])

  useEffect(() => {
    const disconnect = connectRos(ROS, {
      onInfo: (c) => { calib.current = c },
      onFrame: (bmp, stamp, frameId) => { void ingest(bmp, stamp, frameId) },
      onStatus: (text, ok) => setLink({ ok, text }),
      onHealth: (patch) => perception.setHealth(patch),
      onMask: (m) => perception.pushMask(m),
      onDepth: (d) => perception.pushDepth(d),
    })
    return () => {
      perception.reset()
      disconnect()
    }
  }, [ingest])

  const toggle = (k: keyof Layers) => setLayers((l) => ({ ...l, [k]: !l[k] }))

  return (
    <section className="livefeed">
      <header className="livefeed-bar">
        <span className={`dot ${link.ok ? 'ok' : 'bad'}`} />
        <span>{link.ok ? `live camera · ${fps.toFixed(1)} fps` : `camera link: ${link.text || 'down'}`}</span>
        <span className="livefeed-layers">
          {(['mask', 'depth', 'path'] as const).map((k) => (
            <button key={k} type="button" className={layers[k] ? 'on' : ''} aria-pressed={layers[k]} onClick={() => toggle(k)}>
              {k}
            </button>
          ))}
        </span>
      </header>
      <Viewport frame={frame} analysis={analysis} layers={layers} freshness={freshness} />
    </section>
  )
}
