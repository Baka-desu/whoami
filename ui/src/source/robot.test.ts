import { describe, expect, it } from 'vitest'
import { basePoseInMap, composeTf, fresh, invertTf, type RobotState } from './robot'

describe('robot state helpers', () => {
  it('composes map->odom with odom->base_link', () => {
    const p = composeTf({ x: 10, y: 5, yaw: Math.PI / 2 }, { x: 1, y: 0, yaw: 0 })
    expect(p.x).toBeCloseTo(10)
    expect(p.y).toBeCloseTo(6)
    expect(p.yaw).toBeCloseTo(Math.PI / 2)
  })

  it('inverts a transform', () => {
    const a = { x: 3, y: -2, yaw: 0.7 }
    const id = composeTf(a, invertTf(a))
    expect(id.x).toBeCloseTo(0)
    expect(id.y).toBeCloseTo(0)
    expect(id.yaw).toBeCloseTo(0)
  })

  it('treats a stale heartbeat as no signal', () => {
    expect(fresh({ value: true, at: 1000 }, 1500)).toBe(true)
    expect(fresh({ value: true, at: 1000 }, 3000)).toBeUndefined()
    expect(fresh(undefined, 0)).toBeUndefined()
  })

  it('needs both fresh TF links for a base pose', () => {
    const st: RobotState = {
      mapOdom: { value: { x: 0, y: 0, yaw: 0 }, at: 100 },
      odomBase: { value: { x: 1, y: 0, yaw: 0 }, at: 100 },
    }
    expect(basePoseInMap(st, 200)?.x).toBeCloseTo(1)
    expect(basePoseInMap(st, 5000)).toBeNull()
    expect(basePoseInMap({ mapOdom: st.mapOdom }, 200)).toBeNull()
  })
})
