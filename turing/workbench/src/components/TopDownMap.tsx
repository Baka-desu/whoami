import { useEffect, useRef } from 'react'
import type { Freshness } from '../analysis/freshness'
import { CELL_M, TH, TW, Z_MIN, type Analysis } from '../types'

const S = 8 // px per cell

// Same hue family as the viewport's class colours (types.ts CLASS_RGB), so the ground map can
// never disagree with what the photo overlay is calling free, unknown or hazardous.
const FILL = ['#153a4d', '#2c2c2c', '#ff2a2a'] // 0 free (ice-blue family), 1 inflated/unknown, 2 lethal

export function TopDownMap({ analysis, freshness: fr }: { analysis: Analysis; freshness: Freshness | null }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const unsafe = fr ? !fr.ok : false

  useEffect(() => {
    const ctx = ref.current!.getContext('2d')!
    ctx.fillStyle = '#050505'
    ctx.fillRect(0, 0, TW * S, TH * S)

    for (let gz = 0; gz < TH; gz++) {
      for (let gx = 0; gx < TW; gx++) {
        ctx.fillStyle = FILL[analysis.grid[gz * TW + gx]]
        ctx.fillRect(gx * S, (TH - 1 - gz) * S, S - 1, S - 1)
      }
    }

    const px = (x: number) => (x / CELL_M + (TW - 1) / 2 + 0.5) * S
    const pz = (z: number) => (TH - 1 - (z - Z_MIN) / CELL_M + 0.5) * S
    const unit = analysis.meta.kAssumed ? '~%d m' : '%d m'

    // straight-ahead reference, so a curve in the path reads as a real lateral offset
    ctx.strokeStyle = 'rgba(255,255,255,.15)'
    ctx.setLineDash([2, 5])
    ctx.beginPath()
    ctx.moveTo(px(0), 0)
    ctx.lineTo(px(0), TH * S)
    ctx.stroke()

    ctx.font = '10px "JetBrains Mono", ui-monospace, Consolas, monospace'
    ctx.textBaseline = 'bottom'
    for (const m of [5, 10]) {
      const y = pz(m) - S / 2
      ctx.strokeStyle = 'rgba(255,255,255,.25)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(TW * S, y)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#8a8a8a'
      ctx.fillText(unit.replace('%d', String(m)), 3, y - 2)
    }

    if (analysis.path.length > 1) {
      ctx.beginPath()
      analysis.path.forEach((p, i) => (i ? ctx.lineTo(px(p.x), pz(p.z)) : ctx.moveTo(px(p.x), pz(p.z))))
      ctx.strokeStyle = unsafe ? '#8a8a8a' : '#eafff7'
      ctx.lineWidth = 3
      ctx.lineJoin = 'round'
      ctx.setLineDash(unsafe ? [5, 4] : [])
      ctx.stroke()
      ctx.setLineDash([])
    }

    // robot
    const rx = px(0)
    const ry = TH * S
    ctx.beginPath()
    ctx.moveTo(rx, ry - 14)
    ctx.lineTo(rx - 8, ry)
    ctx.lineTo(rx + 8, ry)
    ctx.closePath()
    ctx.fillStyle = '#000'
    ctx.fill()
    ctx.strokeStyle = unsafe ? '#8a8a8a' : '#86f0cf'
    ctx.lineWidth = 2
    ctx.stroke()
  }, [analysis, unsafe])

  return (
    <div className="topdown-wrap">
      <canvas ref={ref} className="topdown" width={TW * S} height={TH * S} />
      {unsafe && <div className="topdown-flag">{fr?.label}</div>}
    </div>
  )
}
