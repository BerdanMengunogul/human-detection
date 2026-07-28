"""
Phase 1: Human detection with bounding boxes from a Hikvision RTSP stream.

Connects to the camera, runs YOLOv8 on each frame, draws green boxes
around detected people, prints the per-frame human count, and shows
the annotated video with an FPS overlay. Press 'q' to quit.

A background thread continuously reads frames from the RTSP socket and
keeps only the latest one. This prevents the multi-second latency that
builds up when frame grabbing can't keep pace with YOLO inference.

Each person is assigned a persistent "Person-N" ID for the life of the
running session. BoT-SORT (with its own built-in ReID) keeps IDs stable
frame-to-frame, including brief occlusions. On top of that, a session-long
appearance gallery re-matches people by embedding similarity whenever the
tracker mints a *new* track ID, so someone who fully leaves the frame and
comes back later (beyond the tracker's own track_buffer window) is still
recognized as the same Person-N instead of getting a new number.

This module is now a thin CLI entrypoint; the actual detection loop lives
in pipeline.py (see that module for DetectionState and run_detection).
"""

import argparse

from pipeline import run_detection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--web", action="store_true",
        help="Serve the detection feed and stats over a LAN-accessible web dashboard "
             "(FastAPI + uvicorn) instead of a local cv2 window.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host/interface to bind the web dashboard to (default: 127.0.0.1, localhost-only; "
             "use 0.0.0.0 for LAN access).",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for the web dashboard (default: 8000).",
    )
    parser.add_argument(
        "--show-window", action="store_true",
        help="When used with --web, also show the local cv2 window alongside the web dashboard.",
    )
    args = parser.parse_args()
    if args.web:
        from webapp import serve_dashboard
        serve_dashboard(host=args.host, port=args.port, show_window=args.show_window)
    else:
        run_detection(state=None, show_window=True)
