import { useEffect, useRef, type CSSProperties } from 'react'
import * as THREE from 'three'

// Concentric rings of turning glyphs with a pointer-led beam. WebGL; capped pixel ratio, stops
// its loop and releases GPU resources on unmount.

const CURSOR_FOLLOW = 8.5

const GLYPH_ART = [
  ['....', '....', '.##.', '#..#', '.##.', '....'],
  ['....', '.#..', '###.', '.#..', '....', '....'],
  ['....', '#..#', '.##.', '.##.', '#..#', '....'],
  ['....', '####', '....', '####', '....', '....'],
  ['.##.', '#..#', '#..#', '#..#', '#..#', '.##.'],
  ['....', '#.#.', '.##.', '.##.', '#.#.', '....'],
  ['....', '..##', '.##.', '##..', '....', '....'],
  ['....', '##..', '.##.', '..##', '....', '....'],
  ['.#.#', '####', '.#.#', '####', '.#.#', '....'],
  ['....', '.##.', '#..#', '####', '#..#', '....'],
  ['###.', '#..#', '###.', '#..#', '###.', '....'],
  ['..#.', '.##.', '###.', '.##.', '..#.', '....'],
]

const GLYPHS = GLYPH_ART.map((rows) =>
  rows.reduce(
    (bits, row, y) =>
      bits + Array.from(row).reduce((acc, ch, x) => acc + (ch === '#' ? Math.pow(2, x + 4 * y) : 0), 0),
    0,
  ),
)

const DEFAULTS = {
  ink: '#2f7a62',
  lit: '#b9fbe6',
  rings: 18,
  charSize: 3,
  gap: 6,
  spin: 8,
  beam: 11,
  band: 20,
  churn: 20,
  scale: 200,
}

type Config = typeof DEFAULTS

function clamp(v: number, lo: number, hi: number, fallback: number): number {
  const n = typeof v === 'number' && isFinite(v) ? v : fallback
  return Math.max(lo, Math.min(hi, n))
}

function settingsFor(cfg: Config) {
  const scale = clamp(cfg.scale, 20, 200, DEFAULTS.scale) / 100
  return {
    rings: clamp(cfg.rings, 1, 20, DEFAULTS.rings),
    charH: scale * clamp(cfg.charSize, 1, 20, DEFAULTS.charSize) * 0.008,
    gapH: scale * clamp(cfg.gap, 0, 20, DEFAULTS.gap) * 0.006,
    spin: clamp(cfg.spin, 0, 20, DEFAULTS.spin) * 0.018,
    beam: clamp(cfg.beam, 0, 20, DEFAULTS.beam) * 0.025,
    band: clamp(cfg.band, 0, 20, DEFAULTS.band) * 0.16,
    churn: clamp(cfg.churn, 0, 20, DEFAULTS.churn) * 0.55,
  }
}

const QUAD_VERTEX = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`

const RING_FRAGMENT = `
  precision highp float;

  #define GLYPH_COUNT ${GLYPHS.length}
  #define TAU 6.28318530718

  uniform vec2 uResolution;
  uniform vec2 uPointer;
  uniform float uHold;
  uniform float uTime;
  uniform float uChurnTime;
  uniform vec3 uInk;
  uniform vec3 uLit;
  uniform float uRings;
  uniform float uCharH;
  uniform float uGapH;
  uniform float uBeam;
  uniform float uBand;
  uniform float uGlyphs[GLYPH_COUNT];

  varying vec2 vUv;

  float hash1(float n) {
    return fract(sin(n * 127.1 + 0.371) * 43758.5453123);
  }

  float hash2(vec2 v) {
    return fract(sin(dot(v, vec2(127.1, 311.7))) * 43758.5453123);
  }

  float glyphAt(int idx, vec2 g) {
    float bits = 0.0;
    for (int i = 0; i < GLYPH_COUNT; i++) {
      if (i == idx) bits = uGlyphs[i];
    }
    float x = min(floor(g.x * 4.0), 3.0);
    float y = min(floor((1.0 - g.y) * 6.0), 5.0);
    return mod(floor(bits / exp2(x + 4.0 * y)), 2.0);
  }

  void main() {
    vec2 centre = uResolution * 0.5;
    vec2 c = vUv * uResolution - centre;
    float radius = length(c);

    float unit = min(uResolution.x, uResolution.y) * 0.5;
    float pitch = unit * (uCharH + uGapH);
    float fill = uCharH / (uCharH + uGapH);
    float ring = floor(radius / pitch);

    if (ring < 1.0 || ring > uRings) discard;

    float slots = max(6.0, floor(TAU * (ring + 0.5)));
    float seed = hash1(ring);

    float heading = mod(ring, 2.0) < 0.5 ? 1.0 : -1.0;
    float turn = uTime * (0.5 + seed * 1.1) * heading + seed;

    float around = fract(atan(c.y, c.x) / TAU + 0.5 + turn);
    float slot = floor(around * slots);

    float lo = (1.0 - fill) * 0.5;
    vec2 g = (vec2(fract(around * slots), fract(radius / pitch)) - lo) / fill;
    if (g.x < 0.0 || g.x > 1.0 || g.y < 0.0 || g.y > 1.0) discard;

    float churn = floor(uChurnTime + seed * 17.0);
    float pick = hash2(vec2(ring, slot) + churn * 5.13);
    int idx = int(min(floor(pick * float(GLYPH_COUNT)), float(GLYPH_COUNT - 1)));
    if (glyphAt(idx, g) < 0.5) discard;

    vec2 pc = uPointer - centre;

    float toBeam = abs(fract((atan(c.y, c.x) - atan(pc.y, pc.x)) / TAU + 0.5) - 0.5);

    float beam = uBeam > 0.0 ? 1.0 - smoothstep(0.0, uBeam, toBeam) : 0.0;
    float onRing = uBand > 0.0
      ? 1.0 - smoothstep(0.0, uBand, abs(radius - length(pc)) / pitch)
      : 0.0;
    float near = clamp(max(beam, onRing), 0.0, 1.0) * uHold;

    vec3 col = mix(uInk, uLit, near);
    float a = 0.4 + 0.6 * near;

    gl_FragColor = vec4(col * a, a);
  }
`

class RingScene {
  private container: HTMLElement
  private cfg: Config
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera = new THREE.Camera()
  private geometry = new THREE.PlaneGeometry(2, 2)
  private material: THREE.ShaderMaterial
  private mesh: THREE.Mesh

  private target = new THREE.Vector2(-1e4, -1e4)
  private eased = new THREE.Vector2(-1e4, -1e4)
  private hold = 0
  private wantHold = 0
  private time = 0
  private churnTime = 0

  private width = 1
  private height = 1
  private frameId = 0
  private lastT = 0
  private disposed = false

  constructor(container: HTMLElement, cfg: Config) {
    this.container = container
    this.cfg = cfg
    const S = settingsFor(cfg)

    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.setClearColor(0x000000, 0)
    const el = this.renderer.domElement
    el.style.position = 'absolute'
    el.style.inset = '0'
    el.style.width = '100%'
    el.style.height = '100%'
    el.style.touchAction = 'none'
    container.appendChild(el)

    this.material = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERTEX,
      fragmentShader: RING_FRAGMENT,
      uniforms: {
        uResolution: { value: new THREE.Vector2(1, 1) },
        uPointer: { value: new THREE.Vector2(-1e4, -1e4) },
        uHold: { value: 0 },
        uTime: { value: 0 },
        uChurnTime: { value: 0 },
        uInk: { value: new THREE.Color(cfg.ink) },
        uLit: { value: new THREE.Color(cfg.lit) },
        uRings: { value: S.rings },
        uCharH: { value: S.charH },
        uGapH: { value: S.gapH },
        uBeam: { value: S.beam },
        uBand: { value: S.band },
        uGlyphs: { value: GLYPHS },
      },
      transparent: true,
      depthTest: false,
      depthWrite: false,
    })

    this.mesh = new THREE.Mesh(this.geometry, this.material)
    this.mesh.frustumCulled = false
    this.scene.add(this.mesh)

    el.addEventListener('pointermove', this.onPointerMove)
    el.addEventListener('pointerdown', this.onPointerMove)
    el.addEventListener('pointerleave', this.onPointerLeave)
    el.addEventListener('pointercancel', this.onPointerLeave)
  }

  private onPointerMove = (e: PointerEvent) => {
    const rect = this.renderer.domElement.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return
    const x = ((e.clientX - rect.left) / rect.width) * this.width
    const y = (1 - (e.clientY - rect.top) / rect.height) * this.height
    this.target.set(x, y)
    if (this.wantHold === 0) this.eased.copy(this.target)
    this.wantHold = 1
  }

  private onPointerLeave = () => {
    this.wantHold = 0
  }

  start() {
    this.lastT = performance.now()
    const loop = () => {
      this.frameId = requestAnimationFrame(loop)
      this.step()
    }
    loop()
  }

  setSize(width: number, height: number) {
    if (this.disposed || width <= 0 || height <= 0) return
    this.renderer.setSize(width, height, false)
    const dpr = this.renderer.getPixelRatio()
    this.width = width * dpr
    this.height = height * dpr
    this.material.uniforms.uResolution.value.set(this.width, this.height)
  }

  updateConfig(cfg: Config) {
    if (this.disposed) return
    this.cfg = cfg
    const u = this.material.uniforms
    u.uInk.value.set(cfg.ink || DEFAULTS.ink)
    u.uLit.value.set(cfg.lit || DEFAULTS.lit)
  }

  private step() {
    if (this.disposed) return
    const now = performance.now()
    let dt = (now - this.lastT) / 1000
    this.lastT = now
    if (!isFinite(dt) || dt < 0) dt = 0
    if (dt > 0.05) dt = 0.05

    const S = settingsFor(this.cfg)
    this.time += dt * S.spin
    this.churnTime += dt * S.churn
    this.eased.lerp(this.target, 1 - Math.exp(-dt * CURSOR_FOLLOW))
    this.hold += (this.wantHold - this.hold) * (1 - Math.exp(-dt * 5))

    const u = this.material.uniforms
    u.uTime.value = this.time
    u.uChurnTime.value = this.churnTime
    u.uPointer.value.copy(this.eased)
    u.uHold.value = this.hold
    u.uRings.value = S.rings
    u.uCharH.value = S.charH
    u.uGapH.value = S.gapH
    u.uBeam.value = S.beam
    u.uBand.value = S.band

    this.renderer.render(this.scene, this.camera)
  }

  dispose() {
    this.disposed = true
    cancelAnimationFrame(this.frameId)
    const el = this.renderer.domElement
    el.removeEventListener('pointermove', this.onPointerMove)
    el.removeEventListener('pointerdown', this.onPointerMove)
    el.removeEventListener('pointerleave', this.onPointerLeave)
    el.removeEventListener('pointercancel', this.onPointerLeave)
    this.geometry.dispose()
    this.material.dispose()
    this.renderer.dispose()
    if (el.parentNode === this.container) this.container.removeChild(el)
  }
}

export interface GlyphRingProps extends Partial<Config> {
  style?: CSSProperties
}

export default function GlyphRing(props: GlyphRingProps) {
  const { style, ...rest } = props
  const containerRef = useRef<HTMLDivElement | null>(null)
  const sceneRef = useRef<RingScene | null>(null)
  const cfg: Config = { ...DEFAULTS, ...rest }
  const cfgRef = useRef<Config>(cfg)
  cfgRef.current = cfg

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let scene: RingScene
    try {
      scene = new RingScene(container, cfgRef.current)
    } catch {
      return // no WebGL: render nothing, the caller's fallback stays visible
    }
    sceneRef.current = scene
    scene.setSize(container.clientWidth, container.clientHeight)
    scene.start()

    const ro = new ResizeObserver(() => scene.setSize(container.clientWidth, container.clientHeight))
    ro.observe(container)
    return () => {
      ro.disconnect()
      scene.dispose()
      sceneRef.current = null
    }
  }, [])

  const { ink, lit } = cfg
  useEffect(() => {
    sceneRef.current?.updateConfig(cfgRef.current)
  }, [ink, lit])

  return (
    <div
      ref={containerRef}
      role="img"
      aria-label="Concentric rings of characters turning under a pointer-led beam"
      style={{ position: 'relative', width: '100%', height: '100%', minWidth: 120, minHeight: 120, overflow: 'hidden', ...style }}
    />
  )
}
