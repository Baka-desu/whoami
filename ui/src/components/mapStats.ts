import type { StatValue } from '../source/api'

// Pure value formatting for the map widget (kept out of StatusWidgets.tsx so that file only exports components).
// Map stats are a flat bag whose keys may be absent, null or (from a misbehaving node) not numbers.
type Stats = Record<string, StatValue>

const MISSING = '—'

const stat = (s: Stats | undefined, key: string): number | null => {
  const v = s?.[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}
const count = (v: number | null) => (v === null ? MISSING : Math.round(v).toLocaleString('en-US'))
const fixed = (v: number | null, unit: string) => (v === null ? MISSING : `${v.toFixed(1)} ${unit}`)
const megabytes = (bytes: number | null) => (bytes === null ? MISSING : fixed(bytes / 1e6, 'MB'))

export function mapRows(s: Stats | undefined): { k: string; v: string }[] {
  const rows = [
    { k: 'keyframes', v: count(stat(s, 'keyframes')) },
    { k: 'loop closures', v: count(stat(s, 'loop_closures')) },
    { k: 'path length', v: fixed(stat(s, 'path_length_m'), 'm') },
    { k: 'cloud points', v: count(stat(s, 'cloud_source_points')) },
    { k: 'elevation cells', v: count(stat(s, 'elevation_known_cells')) },
    { k: 'database', v: megabytes(stat(s, 'db_bytes')) },
    { k: 'depth rate', v: fixed(stat(s, 'depth_hz'), 'Hz') },
    { k: 'last update', v: fixed(stat(s, 'last_update_age_s'), 's') },
  ]
  const mode = s?.mode
  if (typeof mode === 'string' && mode !== '') rows.push({ k: 'mode', v: mode.toUpperCase() })
  return rows
}

export const placeholderCalibration = (s: Stats | undefined) => s?.calibration_placeholder === true
