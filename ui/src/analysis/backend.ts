// Client for the local REST backend (server/): real segmentation + real metric depth,
// run through the exact same geometry pipeline the mock uses (analysis/perception.ts).
import { GH, GW, type Analysis, type FrameMeta } from '../types'
import { buildAnalysis } from './perception'

export interface BackendHealth {
  online: boolean
  seg?: string
  depth?: string
}

export async function checkBackend(url: string): Promise<BackendHealth> {
  try {
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2000) })
    if (!res.ok) return { online: false }
    const j = await res.json()
    return { online: j.status === 'ok', seg: j.seg_model, depth: j.depth_model }
  } catch {
    return { online: false }
  }
}

export async function analyzeViaBackend(frame: ImageBitmap, meta: FrameMeta, url: string): Promise<Analysis> {
  const t0 = performance.now()
  const canvas = document.createElement('canvas')
  canvas.width = frame.width
  canvas.height = frame.height
  canvas.getContext('2d')!.drawImage(frame, 0, 0)
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('could not encode frame'))), 'image/jpeg', 0.85)
  })

  const form = new FormData()
  form.append('file', blob, 'frame.jpg')
  const res = await fetch(`${url}/analyze`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`backend returned ${res.status}`)
  const j = await res.json()

  const mask = base64ToBytes(j.mask_b64)
  const depthBytes = base64ToBytes(j.depth_b64)
  if (mask.length !== GW * GH || depthBytes.length !== GW * GH * 4) {
    throw new Error('backend returned an unexpected payload size')
  }
  const depth = new Float32Array(depthBytes.buffer)

  return buildAnalysis(meta, mask, depth, 'real', performance.now() - t0, { seg: j.seg_model, depth: j.depth_model })
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}
