"""RTSP capture: opening the stream and background-thread reconnect loop."""

import os
import sys
import threading
import time

import cv2

RTSP_URL = os.environ.get("CAMERA_RTSP_URL")
if not RTSP_URL:
    sys.exit(
        "[ERROR] CAMERA_RTSP_URL environment variable is not set. "
        "Set it in your .env file or environment before starting the detector."
    )


def open_stream(exit_on_failure=True):
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[ERROR] Could not connect to RTSP stream: {RTSP_URL}")
        print("Check that the IP, port, credentials, and channel are correct,")
        print("and that the camera is reachable on the network.")
        if exit_on_failure:
            sys.exit(1)
    return cap


class LatestFrameReader:
    """Reads frames from a VideoCapture on a background thread and
    always exposes only the most recently read frame, discarding any
    older ones that the consumer didn't get to in time.

    On a read failure (dropped RTSP socket, camera reboot, network blip)
    the underlying capture is released and reopened with exponential
    backoff (1s, 2s, 5s, 10s, capped at 10s) instead of retrying forever
    against a dead handle.
    """

    RECONNECT_DELAYS = (1, 2, 5, 10)

    def __init__(self, cap, reopen_fn):
        self._cap = cap
        self._reopen_fn = reopen_fn
        self._lock = threading.Lock()
        self._frame = None
        self._ok = False
        self._stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _reconnect(self):
        try:
            self._cap.release()
        except Exception:
            pass
        attempt = 0
        while not self._stopped:
            delay = self.RECONNECT_DELAYS[min(attempt, len(self.RECONNECT_DELAYS) - 1)]
            print(f"[RTSP] Reconnecting in {delay}s (attempt {attempt + 1})...")
            time.sleep(delay)
            if self._stopped:
                return
            try:
                new_cap = self._reopen_fn()
            except Exception as e:
                print(f"[RTSP] Reconnect attempt {attempt + 1} failed: {e}")
                new_cap = None
            if new_cap is not None and new_cap.isOpened():
                self._cap = new_cap
                print("[RTSP] Reconnected successfully.")
                return
            attempt += 1

    def _update(self):
        while not self._stopped:
            ok, frame = self._cap.read()
            with self._lock:
                self._ok = ok
                self._frame = frame
            if not ok:
                self._reconnect()

    def read(self):
        with self._lock:
            return self._ok, self._frame

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=2)
        try:
            self._cap.release()
        except Exception:
            pass
