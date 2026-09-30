// Hardened rosbridge v2 client (JSON over WebSocket) for ROS 2: CompressedImage + CameraInfo.
import type { Intrinsics } from '../types'

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

export interface RosCallbacks {
  onFrame: (bitmap: ImageBitmap, stampMs: number, frameId: string) => void
  onInfo: (calib: CameraCalibration) => void
  onStatus: (text: string, ok: boolean) => void
}

function isValidK(k: unknown): k is number[] {
  if (!Array.isArray(k) || k.length !== 9) return false
  for (let i = 0; i < 9; i++) {
    const val = k[i]
    if (typeof val !== 'number' || !Number.isFinite(val)) return false
  }
  // Intrinsic focal lengths and principal points must be strictly positive
  return k[0] > 0 && k[4] > 0 && k[2] > 0 && k[5] > 0
}

export function connectRos(opts: RosOptions, cb: RosCallbacks): () => void {
  let ws: WebSocket | null = null
  let closed = false
  let reconnectTimer: number | undefined
  let reconnectAttempts = 0

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

      if (!m || typeof m !== 'object' || m.op !== 'publish') return

      if (m.topic === opts.infoTopic) {
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
