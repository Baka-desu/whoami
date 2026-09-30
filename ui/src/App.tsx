import { useCallback, useEffect, useRef, useState } from 'react'
import { analyze } from './analysis/mock'
import { useFreshness } from './analysis/freshness'
import { Inspector } from './components/Inspector'
import { SourcePanel } from './components/SourcePanel'
import { TopBar } from './components/TopBar'
import { Viewport } from './components/Viewport'
import { openCamera } from './source/camera'
import { connectRos, type CameraCalibration, type RosOptions } from './source/rosbridge'
import {
  assumedIntrinsics, type Analysis, type FrameMeta, type Intrinsics, type Layers, type SourceKind, type Status,
} from './types'

const FRAME_INTERVAL_MS = 250

export default function App() {
  const [source, setSource] = useState<SourceKind>('upload')
  const [frame, setFrame] = useState<ImageBitmap | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [layers, setLayers] = useState<Layers>({ image: true, mask: true, depth: false, path: true })
  const [status, setStatus] = useState<Status>('idle')
  const [note, setNote] = useState('')
  const [fps, setFps] = useState(0)
  const [live, setLive] = useState(false)
  const [rosOn, setRosOn] = useState(false)
  const [rosCfg, setRosCfg] = useState<RosOptions>({
    url: 'ws://localhost:9090',
    imageTopic: '/image_raw/compressed',
    infoTopic: '/camera_info',
  })
  const lastFrameAt = useRef(0)
  const rosInfo = useRef<CameraCalibration | null>(null)
  const freshness = useFreshness(analysis)

  const ingest = useCallback((bmp: ImageBitmap, src: SourceKind, stamp: number, frameId: string, streaming: boolean, K?: Intrinsics | null) => {
    const receivedAt = Date.now()
    const meta: FrameMeta = {
      source: src, frameId, stamp, receivedAt, width: bmp.width, height: bmp.height,
      K: K ?? assumedIntrinsics(bmp.width, bmp.height), kAssumed: !K, streaming,
    }
    const now = performance.now()
    const dt = now - lastFrameAt.current
    lastFrameAt.current = now
    setFps((f) => (dt > 0 && dt < 2000 ? f * 0.7 + (1000 / dt) * 0.3 : 0))
    setFrame(bmp)
    setAnalysis(analyze(bmp, meta))
  }, [])

  useEffect(() => () => frame?.close(), [frame])

  const stopLive = () => { setLive(false); setRosOn(false) }
  const afterStop = frame ? 'still' : 'idle'

  const pickSource = (s: SourceKind) => {
    stopLive()
    setSource(s)
    setNote('')
    setStatus(afterStop)
  }

  const upload = async (file: File) => {
    stopLive()
    setNote('')
    try {
      ingest(await createImageBitmap(file), 'upload', Date.now(), file.name, false)
      setStatus('still')
    } catch {
      setStatus('error')
      setNote(`cannot read ${file.name}`)
    }
  }

  const setLiveCamera = (on: boolean) => {
    setLive(on)
    if (!on) setStatus(afterStop)
  }

  const takePhoto = async () => {
    if (live) return setLiveCamera(false)
    try {
      const cam = await openCamera()
      await new Promise((r) => setTimeout(r, 500)) // let exposure settle
      ingest(await createImageBitmap(cam.video), 'camera', Date.now(), 'camera', false)
      cam.stop()
      setStatus('still')
      setNote('')
    } catch (e) {
      setStatus('error')
      setNote(`camera: ${(e as Error).message}`)
    }
  }

  useEffect(() => {
    if (!live) return
    let cancelled = false
    let busy = false
    let timer: number | undefined
    let stop = () => {}
    openCamera().then((cam) => {
      if (cancelled) return cam.stop()
      stop = cam.stop
      setStatus('live')
      setNote('')
      timer = window.setInterval(async () => {
        if (busy) return
        busy = true
        try {
          ingest(await createImageBitmap(cam.video), 'camera', Date.now(), 'camera', true)
        } finally {
          busy = false
        }
      }, FRAME_INTERVAL_MS)
    }).catch((e: Error) => {
      setLive(false)
      setStatus('error')
      setNote(`camera: ${e.message}`)
    })
    return () => {
      cancelled = true
      window.clearInterval(timer)
      stop()
    }
  }, [live, ingest])

  useEffect(() => {
    if (!rosOn) return
    return connectRos(rosCfg, {
      onInfo: (calib) => { rosInfo.current = calib },
      onFrame: (bmp, stamp, frameId) => {
        let resolvedK: Intrinsics | null = null
        const calib = rosInfo.current
        if (calib) {
          const normFrameId = frameId.replace(/^\//, '')
          const normCalibId = calib.frameId.replace(/^\//, '')
          if (!normCalibId || normFrameId === normCalibId) {
            if (calib.width > 0 && calib.height > 0) {
              const scaleX = bmp.width / calib.width
              const scaleY = bmp.height / calib.height
              resolvedK = {
                fx: calib.K.fx * scaleX,
                fy: calib.K.fy * scaleY,
                cx: calib.K.cx * scaleX,
                cy: calib.K.cy * scaleY,
              }
            } else {
              resolvedK = calib.K
            }
          }
        }
        ingest(bmp, 'ros2', stamp, frameId, true, resolvedK)
        setStatus('live')
      },
      onStatus: (text, ok) => {
        setNote(text)
        if (!ok) setStatus('error')
      },
    })
    // rosCfg is locked while connected
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rosOn, ingest])

  const toggleRos = (on: boolean) => {
    setRosOn(on)
    if (!on) {
      rosInfo.current = null
      setNote('')
      setStatus(afterStop)
    }
  }

  return (
    <div className="app">
      <TopBar status={status} fps={fps} note={note} />
      <SourcePanel
        source={source} onSource={pickSource}
        layers={layers} onLayers={setLayers}
        live={live} onLive={setLiveCamera} onPhoto={takePhoto} onUpload={upload}
        rosCfg={rosCfg} onRosCfg={setRosCfg} rosOn={rosOn} onRosOn={toggleRos}
      />
      <Viewport frame={frame} analysis={analysis} layers={layers} freshness={freshness} />
      <Inspector analysis={analysis} freshness={freshness} />
    </div>
  )
}
