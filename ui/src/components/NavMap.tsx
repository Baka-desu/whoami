import { useEffect, useRef } from 'react'
import { fresh, invertTf, composeTf, type RobotSnapshot, type Tf2D } from '../source/robot'

const SIZE = 240
const MAX_AGE_MS = 2000

// Ground-fixed costmap around the robot, heading up. Uses Dev 4's local costmap (Nav2 values:
// -1 unknown, 0 free, 100 lethal), Nav2's /plan, and Dev 2's TF to put everything in one frame.
export function NavMap({ robot }: { robot: RobotSnapshot }) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const ctx = ref.current!.getContext('2d')!
    ctx.fillStyle = '#050505'
    ctx.fillRect(0, 0, SIZE, SIZE)

    const cm = fresh(robot.costmap, robot.now, MAX_AGE_MS)
    const mapOdom = fresh(robot.mapOdom, robot.now, MAX_AGE_MS)
    const odomBase = fresh(robot.odomBase, robot.now, MAX_AGE_MS)
    if (!cm || !odomBase) return

    // robot pose in the costmap's own frame (odom for Nav2's local costmap, map otherwise)
    let rob: Tf2D
    let mapToFrame: Tf2D | null
    if (cm.frame === 'odom') { rob = odomBase; mapToFrame = mapOdom ? invertTf(mapOdom) : null }
    else if (cm.frame === 'map' && mapOdom) { rob = composeTf(mapOdom, odomBase); mapToFrame = { x: 0, y: 0, yaw: 0 } }
    else return

    const half = (Math.max(cm.width, cm.height) * cm.resolution) / 2 // metres shown each side of the robot
    const scale = SIZE / 2 / half // px per metre
    const cos = Math.cos(rob.yaw), sin = Math.sin(rob.yaw)

    const img = ctx.createImageData(SIZE, SIZE)
    for (let py = 0; py < SIZE; py++) {
      for (let px = 0; px < SIZE; px++) {
        const right = (px - SIZE / 2) / scale, fwd = (SIZE / 2 - py) / scale
        const wx = rob.x + fwd * cos + right * sin
        const wy = rob.y + fwd * sin - right * cos
        const cx = Math.floor((wx - cm.originX) / cm.resolution)
        const cy = Math.floor((wy - cm.originY) / cm.resolution)
        const o = (py * SIZE + px) * 4
        let r = 5, g = 5, b = 5
        if (cx >= 0 && cy >= 0 && cx < cm.width && cy < cm.height) {
          const v = cm.data[cy * cm.width + cx]
          if (v < 0) { r = 44; g = 44; b = 44 } // unknown: never free
          else if (v >= 99) { r = 255; g = 42; b = 42 } // lethal
          else { const t = v / 98; r = 21 + 200 * t; g = 58 - 20 * t; b = 77 - 40 * t } // free -> inflated
        }
        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)

    const plan = fresh(robot.plan, robot.now, 10_000)
    if (plan && plan.length > 1 && mapToFrame) {
      ctx.beginPath()
      plan.forEach((p, i) => {
        const f = composeTf(mapToFrame, { x: p.x, y: p.y, yaw: 0 })
        const dx = f.x - rob.x, dy = f.y - rob.y
        const fwd = dx * cos + dy * sin, right = dx * sin - dy * cos
        const X = SIZE / 2 + right * scale, Y = SIZE / 2 - fwd * scale
        if (i) ctx.lineTo(X, Y)
        else ctx.moveTo(X, Y)
      })
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.stroke()
    }

    ctx.fillStyle = '#ff2a2a' // robot, pointing up
    ctx.beginPath()
    ctx.moveTo(SIZE / 2, SIZE / 2 - 9)
    ctx.lineTo(SIZE / 2 + 6, SIZE / 2 + 7)
    ctx.lineTo(SIZE / 2 - 6, SIZE / 2 + 7)
    ctx.closePath()
    ctx.fill()
  }, [robot])

  const cm = fresh(robot.costmap, robot.now, MAX_AGE_MS)
  return (
    <div className="topdown-wrap">
      <canvas ref={ref} width={SIZE} height={SIZE} className="topdown" />
      {!cm && <span className="topdown-flag">NO COSTMAP</span>}
    </div>
  )
}
