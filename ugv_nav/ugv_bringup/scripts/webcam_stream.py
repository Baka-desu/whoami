"""Windows host -> Docker camera bridge for profile live_cam (Dev 5).

ROS runs in Docker / WSL, which cannot open a Windows USB webcam. This runs on the Windows host, owns the
webcam, and serves it as an MJPEG HTTP stream; the camera driver in the container opens it with
    device:=http://host.docker.internal:8090/cam.mjpg

Every client gets the newest frame only (no backlog), so a slow or late reader never sees stale frames.
The driver still stamps each frame on arrival (ugv_bringup README).

    python ugv_nav/ugv_bringup/scripts/webcam_stream.py [--index 0] [--width 640 --height 480] [--port 8090]

Needs Windows Python with opencv-python. --index is the DirectShow order
(ffmpeg -hide_banner -list_devices true -f dshow -i dummy). Width/height must match the calibration.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = b"ugvframe"


class Latest:
    """The newest JPEG and a sequence number; readers block until a newer one exists."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0

    def put(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg, self._seq = jpeg, self._seq + 1
            self._cond.notify_all()

    def wait_newer(self, seq: int, timeout: float) -> tuple[int, bytes | None]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq > seq, timeout=timeout)
            return self._seq, (self._jpeg if self._seq > seq else None)


def capture_loop(cap: cv2.VideoCapture, latest: Latest, quality: int, stop: threading.Event) -> None:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    failures = 0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            failures += 1
            if failures == 30:
                print("webcam_stream: camera is not delivering frames", file=sys.stderr)
            time.sleep(0.01)
            continue
        failures = 0
        ok, jpg = cv2.imencode(".jpg", frame, params)
        if ok:
            latest.put(jpg.tobytes())


def make_handler(latest: Latest):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path.split("?")[0] != "/cam.mjpg":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            try:
                while True:
                    seq, jpeg = latest.wait_newer(seq, timeout=2.0)
                    if jpeg is None:
                        continue  # camera stalled; keep the connection, send nothing (silence = dead camera)
                    self.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, fmt: str, *args) -> None:
            print(f"webcam_stream: {self.client_address[0]} {fmt % args}")

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"webcam_stream: cannot open camera index {a.index}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (a.width, a.height):
        print(f"webcam_stream: camera gives {got[0]}x{got[1]}, asked {a.width}x{a.height}", file=sys.stderr)
        cap.release()
        return 1

    latest, stop = Latest(), threading.Event()
    threading.Thread(target=capture_loop, args=(cap, latest, a.quality, stop), daemon=True).start()
    server = ThreadingHTTPServer((a.bind, a.port), make_handler(latest))
    server.daemon_threads = True
    print(f"webcam_stream: camera {a.index} {a.width}x{a.height} -> http://{a.bind}:{a.port}/cam.mjpg")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
