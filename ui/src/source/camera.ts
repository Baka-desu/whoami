export interface Camera {
  video: HTMLVideoElement
  stop: () => void
}

export async function openCamera(): Promise<Camera> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
    audio: false,
  })
  const video = document.createElement('video')
  video.srcObject = stream
  video.muted = true
  video.playsInline = true
  await video.play()
  return { video, stop: () => stream.getTracks().forEach((t) => t.stop()) }
}
