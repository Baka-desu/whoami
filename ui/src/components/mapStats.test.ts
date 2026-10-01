import { describe, expect, it } from 'vitest'
import { mapRows, placeholderCalibration } from './mapStats'

const row = (rows: { k: string; v: string }[], k: string) => rows.find((r) => r.k === k)?.v

describe('map widget values', () => {
  it('formats every known stat', () => {
    const rows = mapRows({
      keyframes: 12, loop_closures: 3, path_length_m: 41.26, cloud_source_points: 1234567, elevation_known_cells: 8800,
      db_bytes: 52_428_800, depth_hz: 3.24, last_update_age_s: 0.51, mode: 'mapping', calibration_placeholder: false,
    })
    expect(rows).toEqual([
      { k: 'keyframes', v: '12' },
      { k: 'loop closures', v: '3' },
      { k: 'path length', v: '41.3 m' },
      { k: 'cloud points', v: '1,234,567' },
      { k: 'elevation cells', v: '8,800' },
      { k: 'database', v: '52.4 MB' },
      { k: 'depth rate', v: '3.2 Hz' },
      { k: 'last update', v: '0.5 s' },
      { k: 'mode', v: 'MAPPING' },
    ])
  })

  it('shows the neutral placeholder for an absent, null or unusable value, never undefined or NaN', () => {
    const blank = mapRows({}).map((r) => r.v)
    expect(blank).toEqual(new Array(8).fill('—'))
    expect(mapRows(undefined).map((r) => r.v)).toEqual(blank)
    expect(mapRows({ keyframes: null, depth_hz: null, db_bytes: null }).map((r) => r.v)).toEqual(blank)
    const junk = mapRows({ keyframes: Number.NaN, path_length_m: Infinity, db_bytes: 'big', depth_hz: true, loop_closures: '3' })
    expect(junk.map((r) => r.v)).toEqual(blank)
    for (const r of [...blank, ...junk.map((j) => j.v)]) expect(r).not.toMatch(/undefined|NaN|Infinity/)
  })

  it('keeps zero, which is a real reading', () => {
    const rows = mapRows({ keyframes: 0, path_length_m: 0, db_bytes: 0, depth_hz: 0, last_update_age_s: 0 })
    expect(row(rows, 'keyframes')).toBe('0')
    expect(row(rows, 'path length')).toBe('0.0 m')
    expect(row(rows, 'database')).toBe('0.0 MB')
    expect(row(rows, 'depth rate')).toBe('0.0 Hz')
    expect(row(rows, 'last update')).toBe('0.0 s')
  })

  it('lists the mode only when the node sent one', () => {
    expect(row(mapRows({}), 'mode')).toBeUndefined()
    expect(row(mapRows({ mode: '' }), 'mode')).toBeUndefined()
    expect(row(mapRows({ mode: null }), 'mode')).toBeUndefined()
    expect(row(mapRows({ mode: 7 }), 'mode')).toBeUndefined()
    expect(row(mapRows({ mode: 'localize' }), 'mode')).toBe('LOCALIZE')
  })

  it('tolerates stats keys it does not know', () => {
    const rows = mapRows({ keyframes: 2, mask_hz: 3.1, depth_errors: 0, brand_new_stat: 'x' })
    expect(rows).toHaveLength(8)
    expect(row(rows, 'keyframes')).toBe('2')
  })

  it('flags a placeholder calibration only when it is exactly true', () => {
    expect(placeholderCalibration({ calibration_placeholder: true })).toBe(true)
    expect(placeholderCalibration({ calibration_placeholder: false })).toBe(false)
    expect(placeholderCalibration({ calibration_placeholder: null })).toBe(false)
    expect(placeholderCalibration({ calibration_placeholder: 'true' })).toBe(false)
    expect(placeholderCalibration({})).toBe(false)
    expect(placeholderCalibration(undefined)).toBe(false)
  })
})
