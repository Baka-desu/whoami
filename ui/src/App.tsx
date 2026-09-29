import { useCallback, useEffect, useRef, useState } from 'react'
import { analyzeViaBackend, checkBackend } from './analysis/backend'
import { useFreshness } from './analysis/freshness'
import { analyze } from './analysis/mock'
import { Inspector } from './components/Inspector'
import { SourcePanel } from './components/SourcePanel'
import { TopBar } from './components/TopBar'
import { Viewport } from './components/Viewport'
import { openCamera } from './source/camera'
import { connectRos, type RosOptions } from './source/rosbridge'
import {
  assumedIntrinsics, type Analysis, type FrameMeta, type Intrinsics, type Layers, type SourceKind, type Status,
} from './types'

const FRAME_INTERVAL_MS = 250
const BACKEND_URL = 'http://127.0.0.1:8008'
const BACKEND_POLL_MS = 5000

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
  const rosInfo = useRef<Intrinsics | null>(null)
  const freshness = useFreshness(analysis)
  const [backendOnline, setBackendOnline] = useState(false)
  // ingest must stay a stable callback (the live-camera effect depends on it), so the backend's
  // on/off state is read from a ref, not from React state, inside it.
  const backendOnlineRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      const h = await checkBackend(BACKEND_URL)
      if (cancelled) return
      backendOnlineRef.current = h.online
      setBackendOnline(h.online)
    }
    poll()
    const id = window.setInterval(poll, BACKEND_POLL_MS)
    return () => { cancelled = true; window.clearInterval(id) }
  }, [])

  const ingest = useCallback(async (bmp: ImageBitmap, src: SourceKind, stamp: number, frameId: string, streaming: boolean, K?: Intrinsics | null) => {
    const meta: FrameMeta = {
      source: src, frameId, stamp, width: bmp.width, height: bmp.height,
      K: K ?? assumedIntrinsics(bmp.width, bmp.height), kAssumed: !K, streaming,
    }
    const now = performance.now()
    const dt = now - lastFrameAt.current
    lastFrameAt.current = now
    setFps((f) => (dt > 0 && dt < 2000 ? f * 0.7 + (1000 / dt) * 0.3 : 0))
    setFrame(bmp)
    if (backendOnlineRef.current) {
      try {
        setAnalysis(await analyzeViaBackend(bmp, meta, BACKEND_URL))
        return
      } catch {
        // backend dropped mid-session; fall back to mock until the next health poll finds it again
        backendOnlineRef.current = false
        setBackendOnline(false)
      }
    }
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
      await ingest(await createImageBitmap(file), 'upload', Date.now(), file.name, false)
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
      await ingest(await createImageBitmap(cam.video), 'camera', Date.now(), 'camera', false)
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
          await ingest(await createImageBitmap(cam.video), 'camera', Date.now(), 'camera', true)
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

  const rosBusy = useRef(false)
  useEffect(() => {
    if (!rosOn) return
    return connectRos(rosCfg, {
      onInfo: (K) => { rosInfo.current = K },
      onFrame: (bmp, stamp, frameId) => {
        // real inference takes far longer than a rosbridge frame interval; drop frames that
        // arrive while one is still being analysed rather than piling up requests
        if (rosBusy.current) { bmp.close(); return }
        rosBusy.current = true
        ingest(bmp, 'ros2', stamp, frameId, true, rosInfo.current).finally(() => { rosBusy.current = false })
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
      setNote('')
      setStatus(afterStop)
    }
  }

  return (
    <div className="app">
      <TopBar status={status} fps={fps} note={note} backendOnline={backendOnline} models={analysis?.models} />
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
