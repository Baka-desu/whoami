// Minimal rosbridge v2 client (JSON over WebSocket) for ROS 2: a CompressedImage topic + CameraInfo.
import type { Intrinsics } from '../types'

export interface RosOptions {
  url: string
  imageTopic: string
  infoTopic: string
}

export interface RosCallbacks {
  onFrame: (bitmap: ImageBitmap, stampMs: number, frameId: string) => void
  onInfo: (K: Intrinsics, width: number, height: number) => void
  onStatus: (text: string, ok: boolean) => void
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
  }
  ws.onerror = () => cb.onStatus(`cannot reach ${opts.url}`, false)
  ws.onclose = () => { if (!closed) cb.onStatus('ROS 2 disconnected', false) }

  ws.onmessage = async (ev) => {
    const m = JSON.parse(ev.data as string)
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
