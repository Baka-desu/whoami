// State of the console's main view area, as pure functions (tested without a DOM): which view is shown (camera |
// map), the map view's persisted layer toggles, and the error state of the slot both views mount in (App.tsx's
// boundary, so a failed map view never takes the e-stop and the rest of the console down with it).

import type { Enabled } from './useMapData'

// ---- the map view's layer toggles -----------------------------------------------------------------
export interface MapToggles {
  cloud: boolean // accumulated map cloud
  live: boolean // current depth scan
  trajectory: boolean
  grid: boolean // cost grid
  images: boolean // the depth and camera panels
}

export type ToggleKey = keyof MapToggles

export const DEFAULT_TOGGLES: MapToggles = { cloud: true, live: true, trajectory: true, grid: true, images: true }

const TOGGLE_KEYS: readonly ToggleKey[] = ['cloud', 'live', 'trajectory', 'grid', 'images']

// What localStorage held, read leniently: every value that is understood is kept (a saved choice always beats a
// default), anything else (absent, wrong type, unparsable text) falls back to its default.
export function parseToggles(raw: string | null): MapToggles {
  let v: unknown = null
  try {
    v = raw ? JSON.parse(raw) : null
  } catch {
    v = null
  }
  const o = typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
  const out = { ...DEFAULT_TOGGLES }
  for (const k of TOGGLE_KEYS) {
    const b = o[k]
    if (typeof b === 'boolean') out[k] = b
  }
  return out
}

// One toggle flipped, from the state it is given (for a functional state update: two clicks in one render both count).
export const toggled = (t: MapToggles, key: ToggleKey): MapToggles => ({ ...t, [key]: !t[key] })

// The layers the data hook fetches: a layer that is toggled off is not requested at all.
export function enabledLayers(t: MapToggles): Enabled {
  return {
    cloud: t.cloud, live: t.live, trajectory: t.trajectory, grid: t.grid, depth: t.images, camera: t.images,
  }
}

// ---- the main view ---------------------------------------------------------------------------------
export type MainView = 'camera' | 'map'

// The saved choice; anything but exactly 'map' is the camera view.
export const parseView = (raw: string | null): MainView => (raw === 'map' ? 'map' : 'camera')

// ---- the view slot's error state --------------------------------------------------------------------
export interface SlotFailure {
  load: boolean // the view's code could not be downloaded (a stale chunk after a redeploy, a network hiccup)
  detail: string
}
export interface SlotState { key: string; failed: SlotFailure | null }

// Messages browsers (and Vite's preload helper) give a dynamic import that could not be fetched.
const LOAD_FAILURE = /dynamically imported module|Importing a module script failed|Unable to preload/i

export const slotState = (key: string): SlotState => ({ key, failed: null })

// getDerivedStateFromError: what failed, in words the fallback can show.
export function slotOnError(error: unknown): Pick<SlotState, 'failed'> {
  const detail = error instanceof Error
    ? error.message || error.name
    : typeof error === 'string' && error ? error : 'unexpected error'
  return { failed: { load: LOAD_FAILURE.test(detail), detail } }
}

// getDerivedStateFromProps: a new key (another view, or another attempt) clears the failure; null = no change.
export function slotOnProps(state: SlotState, key: string): SlotState | null {
  return state.key === key ? null : slotState(key)
}
