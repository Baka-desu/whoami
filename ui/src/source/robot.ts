// Robot-side state the UI reads over rosbridge from ugv_nav (Dev 2) and ugv_navigation (Dev 4).
// Names/types follow ugv_nav/docs/localization/interfaces.md and src/ugv_navigation/DEV4_INTERFACES.md.

export interface Stamped<T> {
  value: T
  at: number // browser ms when received
}

export interface Pose2D {
  x: number
  y: number
  yaw: number
  receivedAt: number
}

export interface Tf2D {
  x: number
  y: number
  yaw: number
}

export interface Costmap {
  frame: string
  width: number
  height: number
  resolution: number
  originX: number
  originY: number
  data: Int8Array // -1 unknown, 0 free .. 100 lethal
}

export interface RobotState {
  poseValid?: Stamped<boolean> // /ugv/pose_valid (Dev 2, 20 Hz)
  locStatus?: Stamped<string> // /ugv/localization_status (on change)
  odomSource?: Stamped<string> // /ugv/localization/odom_source: wheel | visual
  nav2Heartbeat?: Stamped<boolean> // /ugv/nav2_heartbeat (Dev 4, 20 Hz)
  nav2Status?: Stamped<string> // /ugv/nav2_status: ok | <server> not active ...
  safetyStatus?: Stamped<string> // /ugv/safety_status (safety arbiter, on change): `L<level> <NAME>[: reasons]`
  portMeta?: Stamped<{ valid: boolean; ageS: number; scale: number }> // /segmentation/port_meta (Dev 1)
  perceptionDegraded?: Stamped<boolean> // /ugv/perception_degraded (Dev 1)
  odom?: Stamped<{ x: number; y: number; yaw: number; v: number; w: number }> // /odom (odom frame)
  cmdVelNav2?: Stamped<{ v: number; w: number }> // /cmd_vel_nav2 (candidate only; Dev 5 owns /cmd_vel)
  plan?: Stamped<{ x: number; y: number }[]> // /plan (map frame)
  costmap?: Stamped<Costmap> // /local_costmap/costmap
  mapOdom?: Stamped<Tf2D> // /tf map->odom
  odomBase?: Stamped<Tf2D> // /tf odom->base_link
}

// A heartbeat counts only while fresh: its absence means the publisher died (fail-closed).
export const HEARTBEAT_MAX_AGE_MS = 1000

export function fresh<T>(s: Stamped<T> | undefined, now: number, maxAgeMs = HEARTBEAT_MAX_AGE_MS): T | undefined {
  return s && now - s.at <= maxAgeMs ? s.value : undefined
}

export function composeTf(a: Tf2D, b: Tf2D): Tf2D {
  const c = Math.cos(a.yaw), s = Math.sin(a.yaw)
  return { x: a.x + c * b.x - s * b.y, y: a.y + s * b.x + c * b.y, yaw: a.yaw + b.yaw }
}

export function invertTf(a: Tf2D): Tf2D {
  const c = Math.cos(a.yaw), s = Math.sin(a.yaw)
  return { x: -(c * a.x + s * a.y), y: -(-s * a.x + c * a.y), yaw: -a.yaw }
}

// map -> base_link from Dev 2's two TF links, if both are fresh.
export function basePoseInMap(st: RobotState, now: number): Pose2D | null {
  const mo = fresh(st.mapOdom, now), ob = fresh(st.odomBase, now)
  if (!mo || !ob) return null
  const p = composeTf(mo, ob)
  return { ...p, receivedAt: Math.min(st.mapOdom!.at, st.odomBase!.at) }
}

export interface RobotSnapshot extends RobotState {
  now: number // browser ms when this snapshot was taken, for age checks
}
