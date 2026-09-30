// The production analysis seam. App.tsx talks only to an `Analyzer`; the real Dev 1 perception
// backend (REST) plugs in by implementing this interface - no UI change needed. Nothing in here
// fabricates results: with no analyzer connected the answer is "unavailable" (null).
import type { Analysis, FrameMeta } from '../types'

export interface Analyzer {
  // false while no real perception backend is connected
  readonly available: boolean
  // Resolves to the Perception Port result for this frame, or null when it can't be analysed.
  analyze(frame: ImageBitmap, meta: FrameMeta): Promise<Analysis | null>
}

export const unavailableAnalyzer: Analyzer = {
  available: false,
  analyze: async () => null,
}
