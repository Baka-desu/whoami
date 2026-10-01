import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { Pose } from '../source/api'
import type { CloudFrame, ElevationFrame, GridFrame, TrajectoryFrame } from './codec'
import { buildElevationGeometry, buildGridTexture, colorByHeight, colorElevation, type ElevationColorMode } from './geometry'
import {
  cloudBounds, gridQuad, groundGridFor, homeView, liveHeightBand, pointBounds, unionBounds, yawOf,
  type Bounds, type PlanarPose, type View,
} from './scene-math'

// The 3D map (map view): the accumulated map cloud, the live depth scan coloured by height, the elevation mesh, the
// trajectory, the cost grid draped on the ground and the robot pose, over a ground grid. A plain class in the manner
// of glyph-ring's RingScene: made once per mount on its container, fed through setters, released by dispose(). No
// React in here. World frame = the map frame: metres, x forward/east, y left/north, z up.
//
// Rendering is on demand only: one requestAnimationFrame is scheduled when something changed (a camera move, a
// setter, a resize) and nothing runs in between - this GPU is shared with the segmentation and depth networks.
//
// Colour space. The map layers carry their colours as sRGB bytes (cloud RGB, the height ramp, the elevation colours).
// Points and the elevation mesh are drawn by a small ShaderMaterial that writes those bytes, normalised to 0..1,
// straight to the canvas: it includes neither three's colour-space conversion nor tone mapping, and the canvas is the
// sRGB drawing buffer, so every byte is shown as it is (a built-in material would take vertex colours as linear and
// encode them to sRGB on output, which washes them out). Everything else uses built-in materials the normal three.js
// way: hex colours are sRGB and come out as written; the cost-grid texture is tagged SRGBColorSpace, so the sampler
// decodes it and the output encodes it again, and its bytes too are shown as they are.
//
// Draw order. Objects in the opaque pass are drawn in renderOrder order:
//   -1 ground grid lines, no depth write (whatever is drawn later covers them)
//    0 elevation mesh, pushed back a little in depth so points lying on it win the depth test
//    1 cost grid, draped: no depth test, blended (custom blending keeps it in the opaque pass, after the terrain and
//      before the points), so the halo stays visible on uneven terrain and points stand on top of it
//    2 the two clouds, depth tested
//    3 trajectory and robot pose, no depth test: never hidden inside the cloud

export interface LayerVisibility { cloud: boolean; live: boolean; trajectory: boolean; elevation: boolean; grid: boolean }
type SceneLayer = keyof LayerVisibility
const SCENE_LAYERS: readonly SceneLayer[] = ['cloud', 'live', 'trajectory', 'elevation', 'grid']

const ORDER = { ground: -1, elevation: 0, grid: 1, points: 2, overlay: 3 } as const

const BACKGROUND = '#03100c' // --bg
const GROUND_LINE = '#1f4637' // between --line and --line-strong
const TRAJECTORY = '#86f0cf' // --accent
const GRID_LIFT_M = 0.02 // the cost grid sits just above z = 0
const POSE_AXIS_M = 1 // length of the robot's axis triad (x red = forward, y green, z blue, as in RViz)
const POINT_DEFAULT_M = 0.05
const POINT_MIN_PX = 1.5 // CSS pixels; scaled by the device pixel ratio
const POINT_MAX_PX = 12

const POINT_VERTEX = /* glsl */ `
  attribute vec3 aColor;
  uniform float uSizeM;   // point size in metres
  uniform float uPxPerM;  // drawing-buffer pixels per metre at unit view depth
  uniform vec2 uPxRange;  // clamp, in drawing-buffer pixels
  varying vec3 vColor;
  void main() {
    vColor = aColor;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = clamp(uSizeM * uPxPerM / max(-mv.z, 0.001), uPxRange.x, uPxRange.y);
  }
`

const MESH_VERTEX = /* glsl */ `
  attribute vec3 aColor;
  varying vec3 vColor;
  void main() {
    vColor = aColor;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

// sRGB bytes out as they came in: no colour-space conversion, no tone mapping (see the header).
const BYTE_COLOR_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  void main() {
    gl_FragColor = vec4(vColor, 1.0);
  }
`

const pointMaterial = () =>
  new THREE.ShaderMaterial({
    vertexShader: POINT_VERTEX,
    fragmentShader: BYTE_COLOR_FRAGMENT,
    uniforms: {
      uSizeM: { value: POINT_DEFAULT_M },
      uPxPerM: { value: 1 },
      uPxRange: { value: new THREE.Vector2(POINT_MIN_PX, POINT_MAX_PX) },
    },
  })

const pointSizeM = (spacingM: number) => (Number.isFinite(spacingM) && spacingM > 0 ? Math.min(0.5, Math.max(0.01, spacingM)) : POINT_DEFAULT_M)

// Bounds for frustum culling, set once per frame update: the given box (a cloud's header) or one pass over the
// positions, and the sphere around that box (no second pass).
function fitBounds(g: THREE.BufferGeometry, box: Bounds | null): Bounds | null {
  if (box) g.boundingBox = new THREE.Box3(new THREE.Vector3(box.minX, box.minY, box.minZ), new THREE.Vector3(box.maxX, box.maxY, box.maxZ))
  else g.computeBoundingBox()
  const b = g.boundingBox!
  g.boundingSphere = b.getBoundingSphere(new THREE.Sphere())
  if (box) return box
  const lo = pointBounds(b.min.x, b.min.y, b.min.z)
  const hi = pointBounds(b.max.x, b.max.y, b.max.z)
  return lo && hi ? unionBounds([lo, hi]) : null
}

export class MapScene {
  private container: HTMLElement
  private renderer: THREE.WebGLRenderer
  private controls: OrbitControls
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(50, 1, 0.1, 4000)

  private cloudMat = pointMaterial()
  private liveMat = pointMaterial()
  private elevationMat = new THREE.ShaderMaterial({
    vertexShader: MESH_VERTEX,
    fragmentShader: BYTE_COLOR_FRAGMENT,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 2,
  })
  private trajectoryMat = new THREE.LineBasicMaterial({ color: TRAJECTORY, depthTest: false, depthWrite: false, toneMapped: false })
  private gridMat = new THREE.MeshBasicMaterial({
    map: null,
    side: THREE.DoubleSide,
    transparent: false, // stays in the opaque pass, so renderOrder places it (see the header)
    blending: THREE.CustomBlending, // straight alpha: src * a + dst * (1 - a)
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.SrcAlphaFactor,
    blendDst: THREE.OneMinusSrcAlphaFactor,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  })

  private cloud = new THREE.Points(new THREE.BufferGeometry(), this.cloudMat)
  private live = new THREE.Points(new THREE.BufferGeometry(), this.liveMat)
  private elevation = new THREE.Mesh(new THREE.BufferGeometry(), this.elevationMat)
  private trajectory = new THREE.Line(new THREE.BufferGeometry(), this.trajectoryMat)
  private grid = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.gridMat) // unit plane, scaled to the grid
  private gridTexture: THREE.DataTexture | null = null
  private pose = new THREE.AxesHelper(POSE_AXIS_M)
  private ground: THREE.GridHelper | null = null
  private groundKey = ''

  // What is shown, per layer: the frame it was built from (so an unchanged frame is not rebuilt), whether it has
  // anything to draw, and its bounds (ground grid extent and camera framing).
  private frames: { cloud: CloudFrame | null; live: CloudFrame | null; trajectory: TrajectoryFrame | null; grid: GridFrame | null } =
    { cloud: null, live: null, trajectory: null, grid: null }
  private elev: { frame: ElevationFrame | null; mode: ElevationColorMode; cellOfVertex: Uint32Array | null } =
    { frame: null, mode: 'height', cellOfVertex: null }
  private visible: LayerVisibility = { cloud: true, live: true, trajectory: true, elevation: true, grid: true }
  private drawable: Record<SceneLayer, boolean> = { cloud: false, live: false, trajectory: false, elevation: false, grid: false }
  private bounds: Record<SceneLayer, Bounds | null> = { cloud: null, live: null, trajectory: null, elevation: null, grid: null }
  private robot: PlanarPose | null = null
  private robotKey = ''

  private framed = false // the camera has been placed on data (or the operator moved it first); never again on data
  private frameId = 0
  private disposed = false

  // Throws when WebGL cannot start; the caller shows a message instead.
  constructor(container: HTMLElement) {
    this.container = container
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' })
    const el = this.renderer.domElement
    try {
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      this.renderer.setClearColor(BACKGROUND, 1)
      el.style.position = 'absolute'
      el.style.inset = '0'
      el.style.width = '100%'
      el.style.height = '100%'
      el.style.display = 'block'
      el.style.touchAction = 'none'
      container.appendChild(el)

      this.camera.up.set(0, 0, 1) // z up, set before the controls read it
      this.controls = new OrbitControls(this.camera, el)
      this.controls.screenSpacePanning = false // pan along the ground
      this.controls.maxPolarAngle = Math.PI * 0.495 // stay above the ground
      this.controls.minDistance = 0.5
      this.controls.maxDistance = 2000
      this.controls.addEventListener('change', this.invalidate)
      this.controls.addEventListener('start', this.onOperatorMove)
      el.addEventListener('webglcontextrestored', this.invalidate)
    } catch (e) {
      this.renderer.dispose()
      this.renderer.forceContextLoss()
      el.remove()
      throw e
    }

    this.cloud.renderOrder = ORDER.points
    this.live.renderOrder = ORDER.points
    this.elevation.renderOrder = ORDER.elevation
    this.grid.renderOrder = ORDER.grid
    this.trajectory.renderOrder = ORDER.overlay
    this.pose.renderOrder = ORDER.overlay
    const poseMat = this.pose.material as THREE.LineBasicMaterial
    poseMat.depthTest = false
    poseMat.depthWrite = false
    this.pose.visible = false
    for (const o of [this.cloud, this.live, this.elevation, this.trajectory, this.grid]) o.visible = false
    this.scene.add(this.elevation, this.grid, this.cloud, this.live, this.trajectory, this.pose)

    this.applyView(homeView(null, null))
    this.updateGround()
    this.invalidate()
  }

  // ---- data -------------------------------------------------------------------------------------
  // The accumulated map: camera colours from the frame (height colours if it carries none).
  setCloud(frame: CloudFrame | null) {
    if (this.disposed || frame === this.frames.cloud) return
    this.frames.cloud = frame
    this.setPoints('cloud', this.cloud, frame, frame ? (frame.rgb ?? colorByHeight(frame.xyz, ...liveHeightBand(0))) : null)
  }

  // The current depth scan, coloured by height over a fixed band around the robot's base.
  setLive(frame: CloudFrame | null) {
    if (this.disposed || frame === this.frames.live) return
    this.frames.live = frame
    this.setPoints('live', this.live, frame, frame ? colorByHeight(frame.xyz, ...liveHeightBand(this.robot?.z ?? null)) : null)
  }

  // The elevation mesh. A new frame rebuilds the mesh; a new mode alone rewrites only the colour attribute.
  setElevation(frame: ElevationFrame | null, mode: ElevationColorMode) {
    if (this.disposed) return
    const e = this.elev
    if (frame === e.frame) {
      if (mode === e.mode) return
      e.mode = mode
      const colors = this.elevation.geometry.getAttribute('aColor') as THREE.BufferAttribute | undefined
      if (frame && e.cellOfVertex && colors) {
        colors.set(colorElevation(frame, e.cellOfVertex, mode))
        colors.needsUpdate = true
        this.invalidate()
      }
      return
    }
    e.frame = frame
    e.mode = mode
    e.cellOfVertex = null
    const g = new THREE.BufferGeometry()
    let bounds: Bounds | null = null
    if (frame) {
      const mesh = buildElevationGeometry(frame)
      if (mesh.indices.length > 0) {
        g.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3))
        g.setAttribute('aColor', new THREE.BufferAttribute(colorElevation(frame, mesh.cellOfVertex, mode), 3, true))
        g.setIndex(new THREE.BufferAttribute(mesh.indices, 1))
        bounds = fitBounds(g, null)
        e.cellOfVertex = mesh.cellOfVertex
      }
    }
    this.replaceGeometry(this.elevation, g)
    this.layerChanged('elevation', g.index !== null, bounds)
  }

  // A line through the trajectory's positions.
  setTrajectory(frame: TrajectoryFrame | null) {
    if (this.disposed || frame === this.frames.trajectory) return
    this.frames.trajectory = frame
    const g = new THREE.BufferGeometry()
    let bounds: Bounds | null = null
    if (frame && frame.count >= 2) {
      // Poses are x y z qx qy qz qw: an interleaved attribute reads the first three of every seven in place.
      g.setAttribute('position', new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(frame.poses, 7), 3, 0))
      bounds = fitBounds(g, null)
    }
    this.replaceGeometry(this.trajectory, g)
    this.layerChanged('trajectory', frame !== null && frame.count >= 2, bounds)
  }

  // The cost grid as a texture on a quad just above the ground, placed by its origin, resolution and yaw, sampled
  // nearest-neighbour so every cell stays a crisp square.
  setGrid(frame: GridFrame | null) {
    if (this.disposed || frame === this.frames.grid) return
    this.frames.grid = frame
    this.gridTexture?.dispose()
    this.gridTexture = null
    const q = frame ? gridQuad(frame) : null
    const max = this.renderer.capabilities.maxTextureSize
    if (frame && q && frame.width <= max && frame.height <= max) {
      const rgba = buildGridTexture(frame)
      // Row 0 of the texture is grid row 0 (at the origin); with flipY off it lands at v = 0, the quad's origin side.
      const tex = new THREE.DataTexture(
        new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength), frame.width, frame.height, THREE.RGBAFormat, THREE.UnsignedByteType,
      )
      tex.colorSpace = THREE.SRGBColorSpace
      tex.magFilter = THREE.NearestFilter
      tex.minFilter = THREE.NearestFilter
      tex.generateMipmaps = false
      tex.flipY = false
      tex.needsUpdate = true
      this.gridTexture = tex
      this.grid.position.set(q.cx, q.cy, GRID_LIFT_M)
      this.grid.rotation.set(0, 0, q.yaw)
      this.grid.scale.set(q.sizeX, q.sizeY, 1)
    }
    if ((this.gridMat.map === null) !== (this.gridTexture === null)) this.gridMat.needsUpdate = true // with/without a map is another program
    this.gridMat.map = this.gridTexture
    this.layerChanged('grid', this.gridTexture !== null, this.gridTexture && q ? q.bounds : null)
  }

  // The robot (map -> base_link). Hidden while the pose is unavailable.
  setPose(pose: Pose | null) {
    if (this.disposed) return
    const raw = pose?.available ? [pose.x, pose.y, pose.z, pose.qx, pose.qy, pose.qz, pose.qw] : []
    const v = raw.length === 7 && raw.every((n): n is number => n !== null && Number.isFinite(n)) ? raw : null
    const key = v ? v.join(',') : ''
    if (key === this.robotKey) return // unchanged: a robot standing still costs no frames
    this.robotKey = key
    if (v) {
      const [x, y, z, qx, qy, qz, qw] = v
      this.pose.position.set(x, y, z)
      this.pose.quaternion.set(qx, qy, qz, qw).normalize()
      this.robot = { x, y, z, yaw: yawOf(qx, qy, qz, qw) }
    } else {
      this.robot = null
    }
    this.pose.visible = v !== null
    this.updateGround()
    this.invalidate()
  }

  setLayers(visibility: LayerVisibility) {
    if (this.disposed) return
    this.visible = { ...visibility }
    this.applyVisibility()
    this.invalidate()
  }

  // Back to the home view: behind and above the robot (or over the data), as on the first data.
  resetView() {
    if (this.disposed) return
    this.framed = true
    this.applyView(homeView(this.robot, this.dataBounds()))
  }

  resize(width: number, height: number) {
    if (this.disposed || width <= 0 || height <= 0) return
    this.renderer.setSize(width, height, false)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    const dpr = this.renderer.getPixelRatio()
    const pxPerM = (height * dpr) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2))
    for (const m of [this.cloudMat, this.liveMat]) {
      m.uniforms.uPxPerM.value = pxPerM
      ;(m.uniforms.uPxRange.value as THREE.Vector2).set(POINT_MIN_PX * dpr, POINT_MAX_PX * dpr)
    }
    this.invalidate()
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    cancelAnimationFrame(this.frameId)
    this.frameId = 0
    const el = this.renderer.domElement
    this.controls.removeEventListener('change', this.invalidate)
    this.controls.removeEventListener('start', this.onOperatorMove)
    this.controls.dispose()
    el.removeEventListener('webglcontextrestored', this.invalidate)
    for (const o of [this.cloud, this.live, this.elevation, this.trajectory, this.grid]) o.geometry.dispose()
    this.ground?.dispose()
    this.pose.dispose()
    for (const m of [this.cloudMat, this.liveMat, this.elevationMat, this.trajectoryMat, this.gridMat]) m.dispose()
    this.gridTexture?.dispose()
    this.scene.clear()
    this.renderer.dispose()
    this.renderer.forceContextLoss() // release the context now, not whenever the canvas is collected
    if (el.parentNode === this.container) this.container.removeChild(el)
    // drop the frames, which hold the fetched buffers
    this.frames = { cloud: null, live: null, trajectory: null, grid: null }
    this.elev = { frame: null, mode: this.elev.mode, cellOfVertex: null }
  }

  // ---- internals --------------------------------------------------------------------------------
  private invalidate = () => {
    if (this.disposed || this.frameId !== 0) return
    this.frameId = requestAnimationFrame(this.draw)
  }

  private draw = () => {
    this.frameId = 0
    if (this.disposed) return
    this.renderer.render(this.scene, this.camera)
  }

  // The operator took the camera before any data arrived: do not move it when the data comes.
  private onOperatorMove = () => {
    this.framed = true
  }

  private setPoints(layer: 'cloud' | 'live', points: THREE.Points, frame: CloudFrame | null, colors: Uint8Array | null) {
    const g = new THREE.BufferGeometry()
    let bounds: Bounds | null = null
    const has = frame !== null && colors !== null && frame.count > 0
    if (has) {
      // Both are views onto the fetched buffer (xyz) or the frame's own bytes: handed over without a copy.
      g.setAttribute('position', new THREE.BufferAttribute(frame.xyz, 3))
      g.setAttribute('aColor', new THREE.BufferAttribute(colors, 3, true))
      bounds = fitBounds(g, cloudBounds(frame))
      ;(points.material as THREE.ShaderMaterial).uniforms.uSizeM.value = pointSizeM(frame.spacingM)
    }
    this.replaceGeometry(points, g)
    this.layerChanged(layer, has, bounds)
  }

  // The old geometry's GPU buffers are released as the new one takes its place.
  private replaceGeometry(object: THREE.Points | THREE.Mesh | THREE.Line, g: THREE.BufferGeometry) {
    object.geometry.dispose()
    object.geometry = g
  }

  private layerChanged(layer: SceneLayer, drawable: boolean, bounds: Bounds | null) {
    this.drawable[layer] = drawable
    this.bounds[layer] = drawable ? bounds : null
    this.applyVisibility()
    if (layer !== 'live') this.updateGround() // the live scan moves every frame; the grid follows the map instead
    if (!this.framed && drawable) {
      this.framed = true
      this.applyView(homeView(this.robot, this.dataBounds()))
    }
    this.invalidate()
  }

  private applyVisibility() {
    const objects: Record<SceneLayer, THREE.Object3D> = {
      cloud: this.cloud, live: this.live, trajectory: this.trajectory, elevation: this.elevation, grid: this.grid,
    }
    for (const l of SCENE_LAYERS) objects[l].visible = this.visible[l] && this.drawable[l]
  }

  private robotBounds = () => (this.robot ? pointBounds(this.robot.x, this.robot.y, this.robot.z) : null)

  private dataBounds(): Bounds | null {
    return unionBounds([...SCENE_LAYERS.map((l) => this.bounds[l]), this.robotBounds()])
  }

  // Re-sizes / re-centres the ground grid to the map (not the live scan) and the robot; rebuilt only when it changes.
  private updateGround() {
    const g = groundGridFor(unionBounds([this.bounds.cloud, this.bounds.elevation, this.bounds.trajectory, this.bounds.grid, this.robotBounds()]))
    const key = `${g.cx}:${g.cy}:${g.size}:${g.cell}`
    if (key === this.groundKey) return
    this.groundKey = key
    if (this.ground) {
      this.scene.remove(this.ground)
      this.ground.dispose()
    }
    const grid = new THREE.GridHelper(g.size, Math.round(g.size / g.cell), GROUND_LINE, GROUND_LINE)
    grid.rotation.x = Math.PI / 2 // GridHelper lies in x-z; the map's ground is x-y
    grid.position.set(g.cx, g.cy, 0)
    grid.renderOrder = ORDER.ground
    ;(grid.material as THREE.LineBasicMaterial).depthWrite = false
    this.scene.add(grid)
    this.ground = grid
    this.invalidate()
  }

  private applyView(v: View) {
    this.camera.position.set(v.position.x, v.position.y, v.position.z)
    this.controls.target.set(v.target.x, v.target.y, v.target.z)
    this.controls.update() // re-reads the camera against the target and emits 'change', which schedules a frame
    this.invalidate()
  }
}
