// Hardened rosbridge v2 client (JSON over WebSocket) for ROS 2: a CompressedImage topic +
// CameraInfo, plus sending a NavigateToPose action goal (V1 "localized goal" - distance+bearing
// or local x/y from the start pose, GPS-denied, map/local frame only per architecture.md).
import type { Intrinsics } from '../types'

export interface RosOptions {
  url: string
  imageTopic: string
  infoTopic: string
}

// Fixed by architecture.md/interfaces.md (Dev 2's contract), not a per-robot setting like the
// image/CameraInfo topics: Dev 2 publishes this at 20 Hz, false at startup and on any failure.
const POSE_VALID_TOPIC = '/ugv/pose_valid'

export interface Pose2D {
  x: number
  y: number
  yaw: number
  receivedAt: number
}

interface Tf2D { x: number; y: number; yaw: number }

const normFrame = (f: unknown) => (typeof f === 'string' ? f.replace(/^\//, '') : '')

function parseTf(t: any): Tf2D | null {
  const tr = t?.transform?.translation
  const q = t?.transform?.rotation
  if (!tr || !q) return null
  const v = [tr.x, tr.y, q.x, q.y, q.z, q.w]
  if (!v.every((n) => typeof n === 'number' && Number.isFinite(n))) return null
  return { x: tr.x, y: tr.y, yaw: Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)) }
}

function composeTf(a: Tf2D, b: Tf2D): Tf2D {
  const c = Math.cos(a.yaw), s = Math.sin(a.yaw)
  return { x: a.x + c * b.x - s * b.y, y: a.y + s * b.x + c * b.y, yaw: a.yaw + b.yaw }
}

export interface GoalPose {
  x: number
  y: number
  yawRad: number
  frameId: string
}

export type SendGoal = (pose: GoalPose) => void

export interface CameraCalibration {
  K: Intrinsics
  width: number
  height: number
  frameId: string
}

export interface RosCallbacks {
  onFrame: (bitmap: ImageBitmap, stampMs: number, frameId: string) => void
  onInfo: (calib: CameraCalibration) => void
  onStatus: (text: string, ok: boolean) => void
  // Fired once the socket is open, handing back a function to send a goal on this same
  // connection. Not fired (and the handle goes stale) once the socket closes.
  onReady?: (sendGoal: SendGoal) => void
  // Action feedback/result for a sent goal, if the connected rosbridge/Nav2 reports it.
  onGoalUpdate?: (text: string) => void
  // Dev 2's localization health heartbeat (architecture.md §10.1 / interfaces.md). A localized
  // goal must not be sent while this is false or missing - archV1.md §9/§11 requires a valid
  // reference pose before the conversion to a map-frame goal means anything.
  onPoseValid?: (valid: boolean) => void
  // map -> base_link, composed from Dev 2's /tf (map->odom from RTAB-Map, odom->base_link from
  // the odom selector). Used to freeze the mission reference pose (archV1.md §9).
  onBasePose?: (pose: Pose2D) => void
}

function isValidK(k: unknown): k is number[] {
  if (!Array.isArray(k) || k.length !== 9) return false
  for (let i = 0; i < 9; i++) {
    const val = k[i]
    if (typeof val !== 'number' || !Number.isFinite(val)) return false
  }
  // Intrinsic focal lengths and principal points must be strictly positive; a zero/fake K
  // (e.g. an uncalibrated CameraInfo publisher) must never be accepted as "measured".
  return k[0] > 0 && k[4] > 0 && k[2] > 0 && k[5] > 0
}

export function connectRos(opts: RosOptions, cb: RosCallbacks): () => void {
  let ws: WebSocket | null = null
  let closed = false
  let reconnectTimer: number | undefined
  let reconnectAttempts = 0

  let mapOdom: Tf2D | null = null
  let odomBase: Tf2D | null = null
  let latestProcessedStamp = -Infinity
  let latestDispatchedSeq = 0
  let msgSeq = 0

  const scheduleReconnect = () => {
    if (closed || reconnectTimer !== undefined) return
    reconnectAttempts++
    const delay = Math.min(1000 * Math.pow(1.5, reconnectAttempts - 1), 10000)
    cb.onStatus(`reconnecting to ${opts.url} in ${(delay / 1000).toFixed(1)}s...`, false)
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = undefined
      if (!closed) setupWs()
    }, delay)
  }

  const setupWs = () => {
    if (closed) return
    try {
      ws = new WebSocket(opts.url)
    } catch {
      cb.onStatus(`cannot reach ${opts.url}`, false)
      scheduleReconnect()
      return
    }

    ws.onopen = () => {
      reconnectAttempts = 0
      cb.onStatus(`ROS 2 connected ${opts.url}`, true)
      ws?.send(JSON.stringify({
        op: 'subscribe', topic: opts.imageTopic, type: 'sensor_msgs/msg/CompressedImage',
        throttle_rate: 200, queue_length: 1,
      }))
      ws?.send(JSON.stringify({ op: 'subscribe', topic: opts.infoTopic, type: 'sensor_msgs/msg/CameraInfo' }))
      ws?.send(JSON.stringify({ op: 'subscribe', topic: POSE_VALID_TOPIC, type: 'std_msgs/msg/Bool' }))
      ws?.send(JSON.stringify({ op: 'subscribe', topic: '/tf', type: 'tf2_msgs/msg/TFMessage', throttle_rate: 100 }))
      cb.onReady?.((pose) => ws && sendNavigateToPoseGoal(ws, pose))
    }

    ws.onerror = () => {
      cb.onStatus(`cannot reach ${opts.url}`, false)
    }

    ws.onclose = () => {
      if (!closed) {
        cb.onStatus('ROS 2 disconnected', false)
        scheduleReconnect()
      }
    }

    ws.onmessage = async (ev) => {
      let m: any
      try {
        m = typeof ev.data === 'string' ? JSON.parse(ev.data) : JSON.parse(new TextDecoder().decode(ev.data))
      } catch {
        return
      }
      if (!m || typeof m !== 'object') return

      if (m.op === 'action_feedback' && m.action === '/navigate_to_pose') {
        cb.onGoalUpdate?.(`feedback: ${JSON.stringify(m.values ?? m.feedback ?? {})}`)
        return
      }
      if (m.op === 'action_result' && m.action === '/navigate_to_pose') {
        cb.onGoalUpdate?.(m.values?.result ? 'goal reached' : `result: ${JSON.stringify(m.values ?? {})}`)
        return
      }
      if (m.op !== 'publish') return

      if (m.topic === '/tf') {
        const list = Array.isArray(m.msg?.transforms) ? m.msg.transforms : []
        for (const t of list) {
          const parent = normFrame(t?.header?.frame_id)
          const child = normFrame(t?.child_frame_id)
          const tf = parseTf(t)
          if (!tf) continue
          if (parent === 'map' && child === 'odom') mapOdom = tf
          else if (parent === 'odom' && child === 'base_link') odomBase = tf
        }
        if (mapOdom && odomBase) cb.onBasePose?.({ ...composeTf(mapOdom, odomBase), receivedAt: Date.now() })
      } else if (m.topic === POSE_VALID_TOPIC) {
        const msg = m.msg
        if (msg && typeof msg === 'object' && typeof msg.data === 'boolean') cb.onPoseValid?.(msg.data)
      } else if (m.topic === opts.infoTopic) {
        const msg = m.msg
        if (!msg || typeof msg !== 'object') return
        const width = typeof msg.width === 'number' && Number.isFinite(msg.width) ? msg.width : 0
        const height = typeof msg.height === 'number' && Number.isFinite(msg.height) ? msg.height : 0
        if (width <= 0 || height <= 0 || !isValidK(msg.k)) return
        const frameId = typeof msg.header?.frame_id === 'string' ? msg.header.frame_id : ''
        const k = msg.k as number[]
        cb.onInfo({
          K: { fx: k[0], cx: k[2], fy: k[4], cy: k[5] },
          width,
          height,
          frameId,
        })
      } else if (m.topic === opts.imageTopic) {
        const msg = m.msg
        if (!msg || typeof msg !== 'object') return
        const header = msg.header
        const stamp = header?.stamp
        if (!stamp || typeof stamp.sec !== 'number' || typeof stamp.nanosec !== 'number') return
        const stampMs = stamp.sec * 1000 + stamp.nanosec / 1e6
        const frameId = typeof header.frame_id === 'string' ? header.frame_id : ''
        const data = msg.data
        const format = typeof msg.format === 'string' ? msg.format : ''

        if (typeof data !== 'string' || !data) return
        // Drop frames older than the newest frame processed
        if (stampMs <= latestProcessedStamp) return

        const seq = ++msgSeq
        try {
          const binary = atob(data)
          const bytes = new Uint8Array(binary.length)
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i)
          }
          const isPng = format.toLowerCase().includes('png')
          const blob = new Blob([bytes], { type: isPng ? 'image/png' : 'image/jpeg' })
          const bitmap = await createImageBitmap(blob)

          // Discard if a newer frame already completed decoding or client was closed
          if (stampMs <= latestProcessedStamp || seq < latestDispatchedSeq || closed) {
            bitmap.close()
            return
          }

          latestProcessedStamp = stampMs
          latestDispatchedSeq = seq
          cb.onFrame(bitmap, stampMs, frameId)
        } catch (err) {
          console.warn('Failed to decode ROS 2 frame:', err)
        }
      }
    }
  }

  setupWs()

  return () => {
    closed = true
    if (reconnectTimer !== undefined) {
      window.clearTimeout(reconnectTimer)
      reconnectTimer = undefined
    }
    if (ws) {
      ws.onclose = null
      ws.onerror = null
      ws.onmessage = null
      ws.onopen = null
      ws.close()
      ws = null
    }
  }
}

// rosbridge_suite's native action protocol (v3+): a JSON goal call over the same websocket used
// for topics, no custom ROS node needed on this side. Untested against a live rosbridge/Nav2 -
// there is no ROS 2 install on this machine (see ui/README.md).
function sendNavigateToPoseGoal(ws: WebSocket, { x, y, yawRad, frameId }: GoalPose): void {
  const qz = Math.sin(yawRad / 2)
  const qw = Math.cos(yawRad / 2)
  ws.send(JSON.stringify({
    op: 'send_action_goal',
    id: `navigate_to_pose_${Date.now()}`,
    action: '/navigate_to_pose',
    action_type: 'nav2_msgs/action/NavigateToPose',
    args: {
      pose: {
        header: { frame_id: frameId, stamp: { sec: 0, nanosec: 0 } },
        pose: {
          position: { x, y, z: 0 },
          orientation: { x: 0, y: 0, z: qz, w: qw },
        },
      },
    },
    feedback: true,
  }))
}
