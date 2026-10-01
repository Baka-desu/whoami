// Hardened rosbridge v2 client (JSON over WebSocket) for ROS 2. Reads camera + the robot state
// published by ugv_nav (Dev 2) and ugv_navigation (Dev 4), and can send/cancel a NavigateToPose
// goal and assert the operator e-stop. Never publishes /cmd_vel* (architecture.md §3.1).
import type { Intrinsics } from '../types'
import type { Costmap, RobotState, Stamped, Tf2D } from './robot'
import { parseDepth, parseMask, type RosDepth, type RosMask } from './rosimage'

export interface RosOptions {
  url: string
  imageTopic: string
  infoTopic: string
}

export interface CameraCalibration {
  K: Intrinsics
  width: number
  height: number
  frameId: string
}

export interface GoalPose {
  x: number
  y: number
  yawRad: number
  frameId: string
}

export interface RosApi {
  sendGoal: (pose: GoalPose) => void
  cancelGoal: () => void
  // Level-1 operator kill (dev.md §3, /ugv/e_stop). Re-published while asserted; false releases it.
  setEstop: (asserted: boolean) => void
}

export interface RosCallbacks {
  onFrame: (bitmap: ImageBitmap, stampMs: number, frameId: string) => void
  onInfo: (calib: CameraCalibration) => void
  onStatus: (text: string, ok: boolean) => void
  onReady?: (api: RosApi) => void
  onGoalUpdate?: (text: string) => void
  // Incremental robot-state update (heartbeats, odom, plan, costmap, TF). Fires at topic rate;
  // the receiver should throttle re-rendering.
  onState?: (patch: RobotState) => void
  // Dev 1's Perception Port outputs, decoded and validated (see rosimage.ts)
  onMask?: (mask: RosMask) => void
  onDepth?: (depth: RosDepth) => void
}

const NAV_ACTION = '/navigate_to_pose'
const E_STOP_TOPIC = '/ugv/e_stop'
const ESTOP_REPUBLISH_MS = 200
const MAX_COSTMAP_CELLS = 250_000
const MAX_PLAN_POINTS = 1000

const normFrame = (f: unknown) => (typeof f === 'string' ? f.replace(/^\//, '') : '')
const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

function isValidK(k: unknown): k is number[] {
  if (!Array.isArray(k) || k.length !== 9) return false
  for (let i = 0; i < 9; i++) if (!num(k[i])) return false
  // zero / fake intrinsics must never be accepted as "measured"
  return k[0] > 0 && k[4] > 0 && k[2] > 0 && k[5] > 0
}

function yawOf(q: any): number | null {
  if (![q?.x, q?.y, q?.z, q?.w].every(num)) return null
  return Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
}

function parseTf(t: any): Tf2D | null {
  const tr = t?.transform?.translation
  const yaw = yawOf(t?.transform?.rotation)
  if (!num(tr?.x) || !num(tr?.y) || yaw === null) return null
  return { x: tr.x, y: tr.y, yaw }
}

function parseCostmap(msg: any): Costmap | null {
  const info = msg?.info
  const w = info?.width, h = info?.height, res = info?.resolution
  if (!num(w) || !num(h) || !num(res) || w <= 0 || h <= 0 || res <= 0 || w * h > MAX_COSTMAP_CELLS) return null
  if (!Array.isArray(msg.data) || msg.data.length !== w * h) return null
  const ox = info.origin?.position?.x, oy = info.origin?.position?.y
  if (!num(ox) || !num(oy)) return null
  return {
    frame: normFrame(msg.header?.frame_id), width: w, height: h, resolution: res,
    originX: ox, originY: oy, data: Int8Array.from(msg.data as number[]),
  }
}

function parsePlan(msg: any): { x: number; y: number }[] | null {
  if (!Array.isArray(msg?.poses)) return null
  const step = Math.max(1, Math.ceil(msg.poses.length / MAX_PLAN_POINTS))
  const pts: { x: number; y: number }[] = []
  for (let i = 0; i < msg.poses.length; i += step) {
    const p = msg.poses[i]?.pose?.position
    if (num(p?.x) && num(p?.y)) pts.push({ x: p.x, y: p.y })
  }
  return pts
}

export function connectRos(opts: RosOptions, cb: RosCallbacks): () => void {
  let ws: WebSocket | null = null
  let closed = false
  let reconnectTimer: number | undefined
  let reconnectAttempts = 0
  let estopTimer: number | undefined
  let activeGoalId: string | null = null

  let latestProcessedStamp = -Infinity
  let latestDispatchedSeq = 0
  let msgSeq = 0

  const send = (o: unknown) => { if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(o)) }

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

  const sub = (topic: string, type: string, extra: object = {}) =>
    send({ op: 'subscribe', topic, type, ...extra })

  const stamped = <T,>(value: T): Stamped<T> => ({ value, at: Date.now() })

  const api: RosApi = {
    sendGoal: (pose) => {
      activeGoalId = `navigate_to_pose_${Date.now()}`
      sendNavigateToPoseGoal(send, activeGoalId, pose)
    },
    cancelGoal: () => {
      if (!activeGoalId) return
      send({ op: 'cancel_action_goal', id: activeGoalId, action: NAV_ACTION })
    },
    setEstop: (asserted) => {
      window.clearInterval(estopTimer)
      estopTimer = undefined
      const pub = (data: boolean) => send({ op: 'publish', topic: E_STOP_TOPIC, msg: { data } })
      // latch: a latching publisher lets an arbiter that (re)starts later still receive the last value
      send({ op: 'advertise', topic: E_STOP_TOPIC, type: 'std_msgs/msg/Bool', latch: true })
      pub(asserted)
      if (asserted) estopTimer = window.setInterval(() => pub(true), ESTOP_REPUBLISH_MS)
    },
  }

  const handlePublish = async (m: any) => {
    const msg = m.msg
    if (!msg || typeof msg !== 'object') return
    switch (m.topic) {
      case '/ugv/pose_valid':
        if (typeof msg.data === 'boolean') cb.onState?.({ poseValid: stamped(msg.data) })
        return
      case '/ugv/nav2_heartbeat':
        if (typeof msg.data === 'boolean') cb.onState?.({ nav2Heartbeat: stamped(msg.data) })
        return
      case '/ugv/perception_degraded':
        if (typeof msg.data === 'boolean') cb.onState?.({ perceptionDegraded: stamped(msg.data) })
        return
      case '/ugv/nav2_status':
        if (typeof msg.data === 'string') cb.onState?.({ nav2Status: stamped(msg.data) })
        return
      case '/ugv/safety_status':
        if (typeof msg.data === 'string') cb.onState?.({ safetyStatus: stamped(msg.data) })
        return
      case '/segmentation/port_meta': {
        const d = msg.data
        if (Array.isArray(d) && d.length >= 3 && d.slice(0, 3).every(num)) {
          cb.onState?.({ portMeta: stamped({ valid: d[0] >= 0.5, ageS: d[1], scale: d[2] }) })
        }
        return
      }
      case '/segmentation/mask': {
        const mask = parseMask(msg, Date.now())
        if (mask) cb.onMask?.(mask)
        return
      }
      case '/perception/depth/image': {
        const depth = parseDepth(msg, Date.now())
        if (depth) cb.onDepth?.(depth)
        return
      }
      case '/ugv/localization_status':
        if (typeof msg.data === 'string') cb.onState?.({ locStatus: stamped(msg.data) })
        return
      case '/ugv/localization/odom_source':
        if (typeof msg.data === 'string') cb.onState?.({ odomSource: stamped(msg.data) })
        return
      case '/odom': {
        const p = msg.pose?.pose?.position
        const yaw = yawOf(msg.pose?.pose?.orientation)
        const v = msg.twist?.twist?.linear?.x, w = msg.twist?.twist?.angular?.z
        if (num(p?.x) && num(p?.y) && yaw !== null && num(v) && num(w)) {
          cb.onState?.({ odom: stamped({ x: p.x, y: p.y, yaw, v, w }) })
        }
        return
      }
      case '/cmd_vel_nav2': {
        const v = msg.linear?.x, w = msg.angular?.z
        if (num(v) && num(w)) cb.onState?.({ cmdVelNav2: stamped({ v, w }) })
        return
      }
      case '/plan': {
        const pts = parsePlan(msg)
        if (pts) cb.onState?.({ plan: stamped(pts) })
        return
      }
      case '/local_costmap/costmap': {
        const cm = parseCostmap(msg)
        if (cm) cb.onState?.({ costmap: stamped(cm) })
        return
      }
      case '/tf': {
        const patch: RobotState = {}
        for (const t of Array.isArray(msg.transforms) ? msg.transforms : []) {
          const parent = normFrame(t?.header?.frame_id), child = normFrame(t?.child_frame_id)
          const tf = parseTf(t)
          if (!tf) continue
          if (parent === 'map' && child === 'odom') patch.mapOdom = stamped(tf)
          else if (parent === 'odom' && child === 'base_link') patch.odomBase = stamped(tf)
        }
        if (patch.mapOdom || patch.odomBase) cb.onState?.(patch)
        return
      }
    }

    if (m.topic === opts.infoTopic) {
      const width = num(msg.width) ? msg.width : 0
      const height = num(msg.height) ? msg.height : 0
      if (width <= 0 || height <= 0 || !isValidK(msg.k)) return
      const k = msg.k as number[]
      cb.onInfo({
        K: { fx: k[0], cx: k[2], fy: k[4], cy: k[5] },
        width, height,
        frameId: typeof msg.header?.frame_id === 'string' ? msg.header.frame_id : '',
      })
    } else if (m.topic === opts.imageTopic) {
      const stamp = msg.header?.stamp
      if (!stamp || !num(stamp.sec) || !num(stamp.nanosec)) return
      const stampMs = stamp.sec * 1000 + stamp.nanosec / 1e6
      const frameId = typeof msg.header.frame_id === 'string' ? msg.header.frame_id : ''
      const data = msg.data
      const format = typeof msg.format === 'string' ? msg.format : ''
      if (typeof data !== 'string' || !data) return
      if (stampMs <= latestProcessedStamp) return // older than the newest frame already shown

      const seq = ++msgSeq
      try {
        const binary = atob(data)
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
        const blob = new Blob([bytes], { type: format.toLowerCase().includes('png') ? 'image/png' : 'image/jpeg' })
        const bitmap = await createImageBitmap(blob)
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
      sub(opts.imageTopic, 'sensor_msgs/msg/CompressedImage', { throttle_rate: 200, queue_length: 1 })
      sub(opts.infoTopic, 'sensor_msgs/msg/CameraInfo')
      // ugv_nav (Dev 2)
      sub('/ugv/pose_valid', 'std_msgs/msg/Bool')
      sub('/ugv/localization_status', 'std_msgs/msg/String')
      sub('/ugv/localization/odom_source', 'std_msgs/msg/String')
      sub('/odom', 'nav_msgs/msg/Odometry', { throttle_rate: 100 })
      sub('/tf', 'tf2_msgs/msg/TFMessage', { throttle_rate: 100 })
      // ugv_navigation (Dev 4)
      sub('/ugv/nav2_heartbeat', 'std_msgs/msg/Bool', { throttle_rate: 100 })
      sub('/ugv/nav2_status', 'std_msgs/msg/String')
      // Dev 5 (safety arbiter): what is actually allowed to reach the base, and why
      sub('/ugv/safety_status', 'std_msgs/msg/String')
      sub('/cmd_vel_nav2', 'geometry_msgs/msg/Twist', { throttle_rate: 100 })
      sub('/plan', 'nav_msgs/msg/Path', { throttle_rate: 500 })
      sub('/local_costmap/costmap', 'nav_msgs/msg/OccupancyGrid', { throttle_rate: 500 })
      // Dev 1 Perception Port: degraded flag + the mask / depth the analyzer is built from
      sub('/ugv/perception_degraded', 'std_msgs/msg/Bool')
      sub('/segmentation/port_meta', 'std_msgs/msg/Float64MultiArray')
      sub('/segmentation/mask', 'sensor_msgs/msg/Image', { throttle_rate: 200, queue_length: 1 })
      sub('/perception/depth/image', 'sensor_msgs/msg/Image', { throttle_rate: 500, queue_length: 1 })
      cb.onReady?.(api)
    }

    ws.onerror = () => cb.onStatus(`cannot reach ${opts.url}`, false)

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

      if (m.op === 'action_feedback' && m.action === NAV_ACTION) {
        const f = m.values ?? m.feedback ?? {}
        const d = f.distance_remaining, r = f.number_of_recoveries
        cb.onGoalUpdate?.(num(d) ? `driving: ${d.toFixed(1)} m remaining${num(r) && r > 0 ? `, ${r} recoveries` : ''}` : 'driving')
        return
      }
      if (m.op === 'action_result' && m.action === NAV_ACTION) {
        activeGoalId = null
        const code = m.values?.result?.error_code ?? m.values?.error_code
        const text = m.values?.result?.error_msg ?? m.values?.error_msg
        if (m.result === true) cb.onGoalUpdate?.('goal reached')
        else if (m.status === 5) cb.onGoalUpdate?.('goal cancelled')
        else cb.onGoalUpdate?.(`goal failed${num(code) && code !== 0 ? ` (error ${code}${text ? `: ${text}` : ''})` : ''}`)
        return
      }
      if (m.op === 'publish') await handlePublish(m)
    }
  }

  setupWs()

  return () => {
    closed = true
    window.clearInterval(estopTimer)
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

// rosbridge_suite's native action protocol (v3+): a JSON goal call over the same websocket. Goal
// frame is `map`; Nav2 enforces goal yaw (yaw_goal_tolerance 0.25), so orientation is always set.
function sendNavigateToPoseGoal(send: (o: unknown) => void, id: string, { x, y, yawRad, frameId }: GoalPose): void {
  send({
    op: 'send_action_goal',
    id,
    action: NAV_ACTION,
    action_type: 'nav2_msgs/action/NavigateToPose',
    args: {
      pose: {
        header: { frame_id: frameId, stamp: { sec: 0, nanosec: 0 } },
        pose: {
          position: { x, y, z: 0 },
          orientation: { x: 0, y: 0, z: Math.sin(yawRad / 2), w: Math.cos(yawRad / 2) },
        },
      },
    },
    feedback: true,
  })
}
