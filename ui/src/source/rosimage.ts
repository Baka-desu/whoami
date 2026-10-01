// Strict decoding of the two sensor_msgs/Image messages Dev 1's perception node publishes, as rosbridge
// sends them as JSON: /segmentation/mask (mono8, classes {0,1,2}) and /perception/depth/image (32FC1 metres).
// Anything that does not match the contract exactly is rejected, never guessed at.

export interface RosMask {
  stampMs: number
  frameId: string
  width: number
  height: number
  data: Uint8Array // width*height, as published (values checked later against {0,1,2})
  receivedAt: number
}

export interface RosDepth {
  stampMs: number
  frameId: string
  width: number
  height: number
  data: Float32Array // metres; NaN where there is no measurable depth
  receivedAt: number
}

const MAX_PIXELS = 4_000_000

const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

function bytes(data: unknown): Uint8Array | null {
  if (typeof data === 'string') {
    try {
      const bin = atob(data)
      const out = new Uint8Array(bin.length)
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
      return out
    } catch {
      return null
    }
  }
  if (Array.isArray(data) && data.every((n) => Number.isInteger(n) && n >= 0 && n <= 255)) return Uint8Array.from(data)
  return null
}

function common(msg: any): { stampMs: number; frameId: string; width: number; height: number } | null {
  if (!msg || typeof msg !== 'object') return null
  const s = msg.header?.stamp
  const { width, height } = msg
  if (!num(s?.sec) || !num(s?.nanosec)) return null
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 || width * height > MAX_PIXELS) return null
  return {
    stampMs: s.sec * 1000 + s.nanosec / 1e6,
    frameId: typeof msg.header.frame_id === 'string' ? msg.header.frame_id : '',
    width, height,
  }
}

export function parseMask(msg: any, now: number): RosMask | null {
  const c = common(msg)
  if (!c || msg.encoding !== 'mono8' || msg.step !== c.width) return null
  const data = bytes(msg.data)
  if (!data || data.length !== c.width * c.height) return null
  return { ...c, data, receivedAt: now }
}

export function parseDepth(msg: any, now: number): RosDepth | null {
  const c = common(msg)
  if (!c || msg.encoding !== '32FC1' || msg.step !== c.width * 4) return null
  if (msg.is_bigendian) return null // the little-endian read below would be wrong
  const raw = bytes(msg.data)
  if (!raw || raw.length !== c.width * c.height * 4) return null
  return { ...c, data: new Float32Array(raw.buffer, raw.byteOffset, c.width * c.height), receivedAt: now }
}
