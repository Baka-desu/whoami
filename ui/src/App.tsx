import { useCallback, useEffect, useRef, useState } from 'react'
import { unavailableAnalyzer, type Analyzer } from './analysis/analyzer'
import { useFreshness } from './analysis/freshness'
import { Inspector } from './components/Inspector'
import { SourcePanel } from './components/SourcePanel'
import { TopBar } from './components/TopBar'
import { Viewport } from './components/Viewport'
import { openCamera } from './source/camera'
import { connectRos, type CameraCalibration, type RosApi, type RosOptions } from './source/rosbridge'
import { basePoseInMap, fresh, type Pose2D, type RobotSnapshot, type RobotState } from './source/robot'
import {
  assumedIntrinsics, type Analysis, type FrameMeta, type Intrinsics, type Layers, type SourceKind, type Status,
} from './types'

const FRAME_INTERVAL_MS = 250
const ROBOT_SNAPSHOT_MS = 250 // topics arrive at up to 20 Hz; re-render at 4 Hz
const START_POSE_MAX_AGE_MS = 1000

// Swap for Dev 1's REST analyzer when it exists; nothing else in the UI changes.
const analyzer: Analyzer = unavailableAnalyzer

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
  const ingestSeq = useRef(0)
  const rosApi = useRef<RosApi | null>(null)
  const robotRef = useRef<RobotState>({})
  const [robot, setRobot] = useState<RobotSnapshot | null>(null)
  const [rosConnected, setRosConnected] = useState(false)
  const [goalStatus, setGoalStatus] = useState('')
  const [estop, setEstop] = useState(false)
  const estopRef = useRef(false)
  // Mission reference pose in `map` (archV1.md §9), frozen from Dev 2's TF when the operator sets
  // the start; goals are expressed relative to it, never to wherever the robot is later.
  const [startPose, setStartPose] = useState<Pose2D | null>(null)

  const ingest = useCallback(async (bmp: ImageBitmap, src: SourceKind, stamp: number, frameId: string, streaming: boolean, K?: Intrinsics | null) => {
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
    const seq = ++ingestSeq.current
    let result: Analysis | null = null
    try {
      result = await analyzer.analyze(bmp, meta)
    } catch {
      result = null
    }
    if (seq === ingestSeq.current) setAnalysis(result) // a newer frame already took over
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

  useEffect(() => {
    if (!rosOn) return
    const disconnect = connectRos(rosCfg, {
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
        void ingest(bmp, 'ros2', stamp, frameId, true, resolvedK)
        setStatus('live')
      },
      onStatus: (text, ok) => {
        setNote(text)
        setRosConnected(ok)
        if (!ok) { setStatus('error'); rosApi.current = null }
      },
      onReady: (api) => {
        rosApi.current = api
        if (estopRef.current) api.setEstop(true) // an asserted e-stop survives a reconnect
      },
      onGoalUpdate: setGoalStatus,
      onState: (patch) => { Object.assign(robotRef.current, patch) },
    })
    const snapshotTimer = window.setInterval(() => setRobot({ ...robotRef.current, now: Date.now() }), ROBOT_SNAPSHOT_MS)
    return () => {
      window.clearInterval(snapshotTimer)
      rosApi.current = null
      robotRef.current = {}
      setRobot(null); setRosConnected(false); setStartPose(null)
      disconnect()
    }
    // rosCfg is locked while connected
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rosOn, ingest])

  const toggleRos = (on: boolean) => {
    setRosOn(on)
    if (!on) {
      rosInfo.current = null
      setNote('')
      setGoalStatus('')
      setStatus(afterStop)
    }
  }

  // Goal gate (UI convenience only - Dev 5's safety authority is the real one): never send while
  // the pose is invalid, Nav2 is not heartbeating, perception is degraded, or e-stop is asserted.
  const now = robot?.now ?? 0
  const poseValid = robot ? fresh(robot.poseValid, now) : undefined
  const nav2Up = robot ? fresh(robot.nav2Heartbeat, now) : undefined
  const perceptionDegraded = robot ? fresh(robot.perceptionDegraded, now) : undefined
  const goalBlockedReason = !rosConnected ? ''
    : estop ? 'e-stop asserted'
    : poseValid !== true ? 'pose not valid'
    : nav2Up !== true ? 'Nav2 not active'
    : perceptionDegraded ? 'perception degraded'
    : freshness && !freshness.ok ? freshness.label.toLowerCase()
    : ''
  const canSendGoal = rosConnected && !goalBlockedReason && !!startPose

  const setStart = () => {
    const p = robot ? basePoseInMap(robot, Date.now()) : null
    if (poseValid !== true) return setGoalStatus('cannot set start: pose not valid')
    if (!p || Date.now() - p.receivedAt > START_POSE_MAX_AGE_MS) return setGoalStatus('cannot set start: no fresh map->base_link TF')
    setStartPose(p)
    setGoalStatus('')
  }

  // fwd/left are metres in the start pose's frame (x forward, y left); relYaw is relative to its heading.
  const sendGoal = (fwd: number, left: number, relYaw: number) => {
    const ref = startPose
    if (!rosApi.current || !canSendGoal || !ref) return
    const c = Math.cos(ref.yaw), s = Math.sin(ref.yaw)
    setGoalStatus('goal sent, waiting for feedback...')
    rosApi.current.sendGoal({
      x: ref.x + c * fwd - s * left,
      y: ref.y + s * fwd + c * left,
      yawRad: ref.yaw + relYaw,
      frameId: 'map',
    })
  }

  const toggleEstop = (asserted: boolean) => {
    if (!rosApi.current) return
    rosApi.current.setEstop(asserted)
    estopRef.current = asserted
    setEstop(asserted)
  }

  return (
    <div className="app">
      <TopBar status={status} fps={fps} note={note} analyzerOnline={analyzer.available} />
      <SourcePanel
        source={source} onSource={pickSource}
        layers={layers} onLayers={setLayers}
        live={live} onLive={setLiveCamera} onPhoto={takePhoto} onUpload={upload}
        rosCfg={rosCfg} onRosCfg={setRosCfg} rosOn={rosOn} onRosOn={toggleRos}
        rosConnected={rosConnected} goalStatus={goalStatus}
        onSendGoal={sendGoal} onCancelGoal={() => rosApi.current?.cancelGoal()}
        canSendGoal={canSendGoal} goalBlockedReason={goalBlockedReason}
        hasStart={!!startPose} canSetStart={rosConnected && poseValid === true} onSetStart={setStart}
        estop={estop} onEstop={toggleEstop}
      />
      <Viewport frame={frame} analysis={analysis} layers={layers} freshness={freshness} />
      <Inspector analysis={analysis} freshness={freshness} robot={robot} estop={estop} />
    </div>
  )
}
