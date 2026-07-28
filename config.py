"""Centralized configuration for human_detection.py.

Layered config source, lowest to highest precedence:
  1. Hardcoded defaults below (the values human_detection.py used to have
     as module-level constants - behavior is unchanged if nothing else
     is set).
  2. Optional config.yaml next to this file (or at HD_CONFIG_FILE).
  3. Environment variables (HD_* prefix), highest precedence - convenient
     for one-off overrides (e.g. in a shell or systemd unit) without
     editing a file.

CAMERA_RTSP_URL and DB_PASSWORD are intentionally handled separately
(env-only, via python-dotenv/.env) since they carry live credentials
that must never land in config.yaml or get committed.
"""

import os

import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# (name, type, default) - default is the pre-existing hardcoded value.
_SCHEMA = [
    ("MODEL_NAME", str, "yolov8s.pt"),
    ("REID_MODEL_NAME", str, "yolo26m-reid.onnx"),
    ("TRACKER_CONFIG", str, "tracker_reid.yaml"),
    ("POSE_MODEL_NAME", str, "yolov8s-pose.pt"),
    ("POSE_ENABLED", bool, True),
    ("POSE_EVERY_N_FRAMES", int, 1),

    ("PERSON_CLASS_ID", int, 0),
    ("CONF_THRESHOLD", float, 0.5),
    ("DISPLAY_MAX_WIDTH", int, 1280),
    ("DISPLAY_MAX_HEIGHT", int, 720),

    ("POSE_CONF_THRESHOLD", float, 0.5),
    ("POSE_KEYPOINT_CONF_THRESHOLD", float, 0.5),
    ("POSE_IOU_MATCH_THRESHOLD", float, 0.3),

    ("MIN_BOX_AREA_FRACTION", float, 0.001),
    ("COUNT_SMOOTHING_WINDOW", int, 7),
    ("BOX_SMOOTHING_ALPHA", float, 0.4),
    ("BOX_COAST_FRAMES", int, 5),

    ("REID_MATCH_THRESHOLD", float, 0.4),
    ("REID_TOPUP_INTERVAL_SECONDS", float, 2.0),
    ("REID_MAX_SAMPLES_PER_PERSON", int, 5),

    ("FACE_MATCH_THRESHOLD", float, 0.5),
    ("FACE_MAX_SAMPLES_PER_PERSON", int, 5),
    ("FACE_MIN_BOX_HEIGHT", int, 40),

    ("IDENTIFY_DELAY_SECONDS", float, 0.35),
    ("IDENTIFY_MIN_CANDIDATES", int, 3),

    ("EXIT_GRACE_SECONDS", float, 1.0),

    ("PROFILE_REPORT_INTERVAL", int, 60),

    # Paths: relative values are resolved against the script directory.
    ("ZONES_PATH", str, "door_zones.json"),
    ("ZONES_VERSION_PATH", str, "door_zones.version"),
    ("PEOPLE_PATH", str, "people.json"),
    ("GALLERY_PATH", str, "gallery.npz"),
    ("GALLERY_SAVE_INTERVAL_SECONDS", float, 5.0),

    # Database (PostgreSQL) - DB_PASSWORD is NOT here, see .env.
    ("DB_HOST", str, "localhost"),
    ("DB_PORT", int, 5433),
    ("DB_NAME", str, "human_detection"),
    ("DB_USER", str, "postgres"),
]

_PATH_KEYS = {"ZONES_PATH", "ZONES_VERSION_PATH", "PEOPLE_PATH", "GALLERY_PATH"}


def _coerce(value, py_type):
    if py_type is bool:
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    return py_type(value)


def _load_yaml(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config.yaml must contain a mapping of setting -> value, got {type(data).__name__}")
    return data


class Config:
    pass


def load():
    """Build a Config object: defaults -> config.yaml -> env vars (HD_*)."""
    yaml_path = os.environ.get("HD_CONFIG_FILE", os.path.join(_SCRIPT_DIR, "config.yaml"))
    yaml_values = _load_yaml(yaml_path)

    cfg = Config()
    for name, py_type, default in _SCHEMA:
        value = default
        if name in yaml_values:
            value = _coerce(yaml_values[name], py_type)
        env_name = f"HD_{name}"
        if env_name in os.environ:
            value = _coerce(os.environ[env_name], py_type)
        if name in _PATH_KEYS and not os.path.isabs(str(value)):
            value = os.path.join(_SCRIPT_DIR, value)
        setattr(cfg, name, value)

    cfg.DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
    return cfg
