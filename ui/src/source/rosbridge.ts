// Minimal rosbridge v2 client (JSON over WebSocket) for ROS 2: a CompressedImage topic + CameraInfo,
// plus sending a NavigateToPose action goal (V1 "localized goal" - distance+bearing or local x/y
// from the start pose, GPS-denied, map/local frame only per architecture.md).
import type { Intrinsics } from '../types'

export interface RosOptions {
  url: string
  imageTopic: string
  infoTopic: string
}

export interface GoalPose {
  x: number
  y: number
  yawRad: number
  frameId: string
}

export type SendGoal = (pose: GoalPose) => void

export interface RosCallbacks {
  onFrame: (bitmap: ImageBitmap, stampMs: number, frameId: string) => void
  onInfo: (K: Intrinsics, width: number, height: number) => void
  onStatus: (text: string, ok: boolean) => void
  // Fired once the socket is open, handing back a function to send a goal on this same
  // connection. Not fired (and the handle goes stale) once the socket closes.
  onReady?: (sendGoal: SendGoal) => void
  // Action feedback/result for a sent goal, if the connected rosbridge/Nav2 reports it.
  onGoalUpdate?: (text: string) => void
}

interface RosStamp { sec: number; nanosec: number }

export function connectRos(opts: RosOptions, cb: RosCallbacks): () => void {
  const ws = new WebSocket(opts.url)
  let closed = false

  ws.onopen = () => {
    cb.onStatus(`ROS 2 connected ${opts.url}`, true)
    ws.send(JSON.stringify({
      op: 'subscribe', topic: opts.imageTopic, type: 'sensor_msgs/msg/CompressedImage',
      throttle_rate: 200, queue_length: 1,
    }))
    ws.send(JSON.stringify({ op: 'subscribe', topic: opts.infoTopic, type: 'sensor_msgs/msg/CameraInfo' }))
    cb.onReady?.((pose) => sendNavigateToPoseGoal(ws, pose))
  }
  ws.onerror = () => cb.onStatus(`cannot reach ${opts.url}`, false)
  ws.onclose = () => { if (!closed) cb.onStatus('ROS 2 disconnected', false) }

  ws.onmessage = async (ev) => {
    const m = JSON.parse(ev.data as string)
    if (m.op === 'action_feedback' && m.action === '/navigate_to_pose') {
      cb.onGoalUpdate?.(`feedback: ${JSON.stringify(m.values ?? m.feedback ?? {})}`)
      return
    }
    if (m.op === 'action_result' && m.action === '/navigate_to_pose') {
      cb.onGoalUpdate?.(m.values?.result ? 'goal reached' : `result: ${JSON.stringify(m.values ?? {})}`)
      return
    }
    if (m.op !== 'publish') return
    if (m.topic === opts.infoTopic) {
      const k: number[] = m.msg.k
      cb.onInfo({ fx: k[0], cx: k[2], fy: k[4], cy: k[5] }, m.msg.width, m.msg.height)
    } else if (m.topic === opts.imageTopic) {
      const { header, format, data } = m.msg as { header: { stamp: RosStamp; frame_id: string }; format: string; data: string }
      const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0))
      const blob = new Blob([bytes], { type: format.includes('png') ? 'image/png' : 'image/jpeg' })
      cb.onFrame(await createImageBitmap(blob), header.stamp.sec * 1000 + header.stamp.nanosec / 1e6, header.frame_id)
    }
  }

  return () => { closed = true; ws.close() }
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
