import os
import json
import numpy as np
import torch
from collections import deque
from copy import deepcopy

# Configuration
PROCESSED_FOLDER = 'static/processed/'
DATABASE = 'traffic_multi1.db'
CONFIG_PATH = os.environ.get("ZEBRA_CONFIG_PATH", "config_v2.json")

DEBUG = os.environ.get("ZEBRA_DEBUG", "0") == "1"

DEFAULT_CAMERA_CONFIG = [
    {
        "id": "cam1",
        "name": "Camera 1",
        "rtsp_url": "rtsp://192.168.0.51:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
        "enabled": True,
        "zones": {"zebra": [(300, 461), (2170, 489)]},
    },
    {
        "id": "cam2",
        "name": "Camera 2",
        "rtsp_url": "rtsp://192.168.0.52:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
        "enabled": True,
        "zones": {"zebra": [(546, 484), (997, 472)]},
    },
    {
        "id": "cam3",
        "name": "Camera 3",
        "rtsp_url": "rtsp://192.168.0.53:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
        "enabled": True,
        "zones": {"zebra": [(436, 523), (992, 535)]},
    },
    {
        "id": "cam4",
        "name": "Camera 4",
        "rtsp_url": "rtsp://192.168.0.54:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
        "enabled": True,
        "zones": {"zebra": [(271, 73), (287, 694)]},
    },
    {
        "id": "cam5",
        "name": "Camera 5",
        "rtsp_url": "rtsp://192.168.0.55:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
        "enabled": True,
        "zones": {"zebra": [(300, 461), (2170, 489)]},
    },
]

DEFAULT_PROCESSING_SETTINGS = {
    "frame_skip": 2,  # Changed from 8 to 2
    "vehicle_threshold": 2,
    "image_save_interval": 1,  # Save images every IMAGE_SAVE_INTERVAL seconds
    "process_duration": 2,  # Process each stream for 2 seconds
}

DEFAULT_RUNTIME_SETTINGS = {
    "max_queue_size": 32,
}

def load_runtime_config(path: str = CONFIG_PATH):
    """Load runtime configuration from JSON (or defaults).

    Parameters
    ----------
    path : str
        File path to JSON config. Defaults to CONFIG_PATH env override.

    Returns
    -------
    dict
        - cameras: list of camera definitions {id, name, rtsp_url, enabled, zones}
        - processing: frame_skip, vehicle_threshold, image_save_interval, process_duration
        - runtime: max_queue_size and other execution flags

    Notes
    -----
    Malformed JSON is caught and defaults are used.
    """
    config = {
        "cameras": deepcopy(DEFAULT_CAMERA_CONFIG),
        "processing": deepcopy(DEFAULT_PROCESSING_SETTINGS),
        "runtime": deepcopy(DEFAULT_RUNTIME_SETTINGS),
    }

    if os.path.exists(path):
        try:
            with open(path, "r") as file:
                try:
                    loaded_config = json.load(file)
                    config["cameras"] = loaded_config.get("cameras", config["cameras"])
                    config["processing"].update(loaded_config.get("processing", {}))
                    config["runtime"].update(loaded_config.get("runtime", {}))
                except json.JSONDecodeError:
                    print(f"[WARN] Unable to parse {path}, using defaults.")
        except OSError as exc:
            print(f"[WARN] Unable to read {path}: {exc}. Using defaults.")
    else:
        print(f"[INFO] No runtime config found at {path}, using defaults.")

    # drop malformed cameras missing required fields
    valid_cameras = []
    for cam in config["cameras"]:
        if cam.get("id") and cam.get("rtsp_url"):
            valid_cameras.append(cam)
        elif DEBUG:
            print(f"[DEBUG] Skipping camera entry missing id or rtsp_url: {cam}")
    config["cameras"] = valid_cameras

    return config

CONFIG = load_runtime_config()

FRAME_SKIP = CONFIG["processing"].get("frame_skip", 2)
VEHICLE_THRESHOLD = CONFIG["processing"].get("vehicle_threshold", 2)
IMAGE_SAVE_INTERVAL = CONFIG["processing"].get("image_save_interval", 1)
PROCESS_DURATION = CONFIG["processing"].get("process_duration", 2)
MAX_QUEUE_SIZE = CONFIG["runtime"].get("max_queue_size", 32)

# Authentication settings
DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'admin@123!'

# RTSP URLs for each camera (now dynamic)
RTSP_URLS = {
    cam["id"]: cam["rtsp_url"]
    for cam in CONFIG["cameras"]
    if cam.get("enabled", True) and cam.get("rtsp_url") and cam.get("id")
}

# Model configuration
NUM_BLOCKS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = os.cpu_count()  # Use os.cpu_count() instead of mp.cpu_count()
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

# Color mappings
COLOR_MAPPING = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}

REGIONS_FILE = 'regions_config.json'

# Hardcoded default regions - now hydrated from config to enable dynamic camera counts
def build_default_regions(config):
    regions = {}
    for cam in config["cameras"]:
        cam_id = cam.get("id")
        if not cam.get("enabled", True) or not cam_id:
            continue
        zebra_vertices = cam.get("zones", {}).get("zebra")
        if zebra_vertices:
            regions[cam_id] = {
                "Zebra": {
                    "vertices": np.array(zebra_vertices, dtype=np.int32),
                    "weight": 1.0,
                    "color": (0, 0, 255),
                }
            }
    return regions

DEFAULT_REGIONS = build_default_regions(CONFIG)

PROCESSING_TIMES = {cam_id: deque(maxlen=50) for cam_id in RTSP_URLS.keys()}

# Create necessary directories
def init_directories():
    os.makedirs('static/temp/', exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    for cam in RTSP_URLS.keys():
        os.makedirs(os.path.join(PROCESSED_FOLDER, cam), exist_ok=True)

# Load regions from config file
def load_regions():
    regions = DEFAULT_REGIONS.copy()

    if os.path.exists(REGIONS_FILE):
        with open(REGIONS_FILE, 'r') as file:
            saved_regions = json.load(file)

        for cam, cam_regions in regions.items():
            if cam in saved_regions:
                for r_name, r_data in saved_regions[cam].items():
                    if r_name == 'Zebra' and 'vertices' in r_data:
                        r_data['vertices'] = np.array(r_data['vertices'], dtype=np.int32)
                        regions[cam].update({r_name: r_data})

    return regions

# Configure CUDA settings if available
def configure_cuda():
    if DEBUG:
        device_label = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DEBUG] CUDA available: {torch.cuda.is_available()} | Using device: {device_label}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

# Initialize environment
def initialize_environment():
    init_directories()
    configure_cuda()
    if DEBUG:
        print(f"[DEBUG] Loaded {len(RTSP_URLS)} camera streams from config ({CONFIG_PATH})")
    return load_regions()

REGIONS = initialize_environment()
