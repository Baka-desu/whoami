import { useEffect, useRef } from 'react'
import type { Freshness } from '../analysis/freshness'
import { CLASS_NAMES, CLASS_RGB, GH, GW, PERCEPTION_MAX_AGE_MS, type Analysis, type Layers } from '../types'
import { NavHud } from './NavHud'

const layerCanvas = document.createElement('canvas')
layerCanvas.width = GW
layerCanvas.height = GH
const lctx = layerCanvas.getContext('2d')!

function paintLayer(ctx: CanvasRenderingContext2D, w: number, h: number, rgba: (i: number) => number[]) {
  const img = lctx.createImageData(GW, GH)
  for (let i = 0; i < GW * GH; i++) img.data.set(rgba(i), i * 4)
  lctx.putImageData(img, 0, 0)
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(layerCanvas, 0, 0, w, h)
  ctx.imageSmoothingEnabled = true
}

const MASK_ALPHA = [60, 130, 170]
// Depth uses amber-to-black: a hue no class colour touches, so a mask + depth combination never
// reads as "this is traversable" or "this is hazard" by accident. DEPTH_FLOOR keeps the far end
// dim rather than pitch black, so the photo dropping out never removes all spatial context.
const DEPTH_FLOOR = 0.16

function depthFill(t: number): [number, number, number] {
  const v = DEPTH_FLOOR + (1 - DEPTH_FLOOR) * Math.max(0, Math.min(1, t))
  return [Math.round(70 * v + 15), Math.round(48 * v + 8), Math.round(10 * v)]
}

interface Props {
  frame: ImageBitmap | null
  analysis: Analysis | null
  layers: Layers
  freshness: Freshness | null
}

export function Viewport({ frame, analysis, layers, freshness: fr }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const c = ref.current
    if (!c || !frame) return
    const w = Math.min(frame.width, 960)
    const h = Math.round((w * frame.height) / frame.width)
    c.width = w
    c.height = h
    const ctx = c.getContext('2d')!
    ctx.fillStyle = '#050505'
    ctx.fillRect(0, 0, w, h)
    if (layers.image) ctx.drawImage(frame, 0, 0, w, h)
    if (!analysis) return

    const unsafe = fr ? !fr.ok : false

    if (layers.depth) {
      paintLayer(ctx, w, h, (i) => {
        const t = 1 - (analysis.depth[i] - 1.5) / 13.5
        return [...depthFill(t), unsafe ? 120 : 225]
      })
    }
    if (layers.mask) {
      paintLayer(ctx, w, h, (i) => {
        const cls = analysis.mask[i]
        const baseAlpha = MASK_ALPHA[cls]
        const alpha = unsafe ? Math.round(baseAlpha * 0.3) : baseAlpha
        return [...CLASS_RGB[cls], alpha]
      })
      // Hazard stays legible under any depth fill: a thin outline on top of everything else.
      outlineClass(ctx, w, h, analysis.mask, 2, unsafe)
    }
    if (layers.path && analysis.pathPx.length > 1) {
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'
      ctx.beginPath()
      analysis.pathPx.forEach((p, i) => (i ? ctx.lineTo(p.u * w, p.v * h) : ctx.moveTo(p.u * w, p.v * h)))
      if (unsafe) {
        ctx.setLineDash([8, 6])
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 7
        ctx.stroke()
        ctx.strokeStyle = '#8a8a8a'
        ctx.lineWidth = 4
        ctx.stroke()
        ctx.setLineDash([])
      } else {
        ctx.setLineDash([])
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 9
        ctx.stroke()
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 4
        ctx.stroke()

        const goal = analysis.pathPx[analysis.pathPx.length - 1]
        ctx.beginPath()
        ctx.arc(goal.u * w, goal.v * h, 8, 0, Math.PI * 2)
        ctx.strokeStyle = '#ff2a2a'
        ctx.lineWidth = 3
        ctx.stroke()
      }
    }
  }, [frame, analysis, layers, fr])

  const activeLegend = layers.mask ? CLASS_NAMES.map((n, i) => [i, n] as const) : []

  return (
    <main className="viewport">
      {frame ? (
        <>
          <NavHud analysis={analysis} />
          <canvas ref={ref} />
          {fr && !fr.ok && (
            <div className={`banner ${fr.stale ? 'stale' : 'degraded'}`}>
              {fr.stale ? `STALE · ${fr.ageMs} / ${PERCEPTION_MAX_AGE_MS} ms` : 'PERCEPTION DEGRADED'}
            </div>
          )}
          {analysis?.meta.kAssumed && <div className="assumed">K ASSUMED · NOT METRIC</div>}
          {(activeLegend.length > 0 || layers.path) && (
            <div className="legend">
              {activeLegend.map(([i, n]) => (
                <span key={n}>
                  <i style={{ background: `rgb(${CLASS_RGB[i].join(',')})` }} />
                  {i} {n}
                </span>
              ))}
              {layers.depth && <span><i style={{ background: `rgb(${depthFill(0.8).join(',')})` }} />NEAR</span>}
              {layers.path && <span><i className="path" />PATH</span>}
            </div>
          )}
        </>
      ) : (
        <div className="nosignal">
          <b>NO SIGNAL</b>
          <span>Upload a photo, take one, or start a live source</span>
        </div>
      )}
    </main>
  )
}

// Traces the boundary of every pixel of `cls` with a 1px-in-source outline (scaled up), so the
// shape reads even when the fill underneath is covered by another layer's alpha.
function outlineClass(ctx: CanvasRenderingContext2D, w: number, h: number, mask: Uint8Array, cls: number, unsafe = false) {
  const img = lctx.createImageData(GW, GH)
  const outlineColor = unsafe ? [180, 42, 42, 80] : [255, 42, 42, 255]
  for (let v = 0; v < GH; v++) {
    for (let u = 0; u < GW; u++) {
      const i = v * GW + u
      if (mask[i] !== cls) continue
      const edge = u === 0 || v === 0 || u === GW - 1 || v === GH - 1
        || mask[i - 1] !== cls || mask[i + 1] !== cls || mask[i - GW] !== cls || mask[i + GW] !== cls
      if (edge) img.data.set(outlineColor, i * 4)
    }
  }
  lctx.putImageData(img, 0, 0)
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(layerCanvas, 0, 0, w, h)
  ctx.imageSmoothingEnabled = true
}
