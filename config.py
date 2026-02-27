"""
Configuration module for Zebra Crossing Analytics v2.
Loads all settings from config.json for fully dynamic configuration.
Supports GPU (CUDA/Jetson Orin) with automatic device detection.
"""

import os
import json
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

CONFIG_FILE = os.environ.get('ZC_CONFIG_FILE', 'config.json')
REGIONS_FILE = os.environ.get('ZC_REGIONS_FILE', 'regions_config.json')

# Color name to BGR tuple mapping
COLOR_MAPPING = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}


class AppConfig:
    """Dynamic application configuration loaded from config.json."""

    def __init__(self):
        self._lock = threading.Lock()
        self._config = {}
        self.load()

    def load(self):
        """Load configuration from JSON file."""
        logger.debug("[CONFIG] Loading configuration from %s", CONFIG_FILE)
        try:
            with open(CONFIG_FILE, 'r') as f:
                self._config = json.load(f)
            logger.info("[CONFIG] Configuration loaded successfully")
            logger.debug("[CONFIG] Cameras configured: %s", list(self.cameras.keys()))
            logger.debug("[CONFIG] Device setting: %s", self._config.get('processing', {}).get('device', 'auto'))
        except FileNotFoundError:
            logger.error("[CONFIG] Config file not found: %s", CONFIG_FILE)
            raise
        except json.JSONDecodeError as e:
            logger.error("[CONFIG] Invalid JSON in config file: %s", e)
            raise

    def reload(self):
        """Thread-safe config reload."""
        with self._lock:
            self.load()
            logger.info("[CONFIG] Configuration reloaded")

    def save(self):
        """Save current configuration back to JSON file."""
        with self._lock:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self._config, f, indent=4)
            logger.info("[CONFIG] Configuration saved to %s", CONFIG_FILE)

    # --- Camera properties ---

    @property
    def cameras(self):
        """Return dict of enabled cameras {cam_id: {name, rtsp_url, enabled}}."""
        return {
            k: v for k, v in self._config.get('cameras', {}).items()
            if v.get('enabled', True)
        }

    @property
    def all_cameras(self):
        """Return all cameras including disabled ones."""
        return self._config.get('cameras', {})

    @property
    def camera_ids(self):
        """Return sorted list of enabled camera IDs."""
        return sorted(self.cameras.keys())

    @property
    def num_cameras(self):
        """Return count of enabled cameras."""
        return len(self.cameras)

    def get_rtsp_urls(self):
        """Return dict of {cam_id: rtsp_url} for enabled cameras."""
        return {k: v['rtsp_url'] for k, v in self.cameras.items()}

    def add_camera(self, cam_id, name, rtsp_url, enabled=True):
        """Add a new camera configuration."""
        with self._lock:
            self._config.setdefault('cameras', {})[cam_id] = {
                'name': name,
                'rtsp_url': rtsp_url,
                'enabled': enabled
            }
            logger.info("[CONFIG] Camera added: %s (%s)", cam_id, name)
            self.save()

    def remove_camera(self, cam_id):
        """Remove a camera configuration."""
        with self._lock:
            if cam_id in self._config.get('cameras', {}):
                del self._config['cameras'][cam_id]
                logger.info("[CONFIG] Camera removed: %s", cam_id)
                self.save()

    def update_camera(self, cam_id, **kwargs):
        """Update camera properties."""
        with self._lock:
            if cam_id in self._config.get('cameras', {}):
                self._config['cameras'][cam_id].update(kwargs)
                logger.info("[CONFIG] Camera updated: %s -> %s", cam_id, kwargs)
                self.save()

    # --- Processing properties ---

    @property
    def frame_skip(self):
        return self._config.get('processing', {}).get('frame_skip', 2)

    @property
    def vehicle_threshold(self):
        return self._config.get('processing', {}).get('vehicle_threshold', 2)

    @property
    def process_duration(self):
        return self._config.get('processing', {}).get('process_duration', 2)

    @property
    def image_save_interval(self):
        return self._config.get('processing', {}).get('image_save_interval', 1)

    @property
    def max_queue_size(self):
        return self._config.get('processing', {}).get('max_queue_size', 32)

    @property
    def use_sahi(self):
        return self._config.get('processing', {}).get('use_sahi', True)

    @use_sahi.setter
    def use_sahi(self, value):
        self._config.setdefault('processing', {})['use_sahi'] = value

    @property
    def vehicle_classes(self):
        return self._config.get('processing', {}).get('vehicle_classes',
                                                       ['car', 'truck', 'bus', 'motorcycle', 'bicycle'])

    @property
    def device(self):
        """Determine compute device: auto-detect CUDA/CPU."""
        device_setting = self._config.get('processing', {}).get('device', 'auto')
        if device_setting == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    logger.info("[CONFIG] GPU detected: %s", device_name)
                    return 'cuda'
                else:
                    logger.info("[CONFIG] No GPU detected, using CPU")
                    return 'cpu'
            except ImportError:
                logger.warning("[CONFIG] PyTorch not available, using CPU")
                return 'cpu'
        return device_setting

    # --- Auth properties ---

    @property
    def username(self):
        return self._config.get('auth', {}).get('username', 'admin')

    @property
    def password(self):
        return self._config.get('auth', {}).get('password', 'admin@123!')

    # --- Server properties ---

    @property
    def host(self):
        return self._config.get('server', {}).get('host', '0.0.0.0')

    @property
    def port(self):
        return self._config.get('server', {}).get('port', 3000)

    @property
    def debug(self):
        return self._config.get('server', {}).get('debug', True)

    # --- Logging properties ---

    @property
    def log_level(self):
        return self._config.get('logging', {}).get('level', 'DEBUG')

    @property
    def log_format(self):
        return self._config.get('logging', {}).get('format',
                                                    '%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # --- Paths ---

    @property
    def processed_folder(self):
        return 'static/processed/'

    @property
    def database_path(self):
        return self._config.get('database', {}).get('path', 'traffic_v2.db')

    # --- Regions ---

    def load_regions(self):
        """Load region configurations, merging defaults from config with saved overrides."""
        logger.debug("[CONFIG] Loading region configurations")
        regions = {}

        # Load defaults from config.json
        config_regions = self._config.get('regions', {})
        for cam_id, cam_regions in config_regions.items():
            regions[cam_id] = {}
            for region_name, region_data in cam_regions.items():
                regions[cam_id][region_name] = {
                    'vertices': np.array(region_data['vertices'], dtype=np.int32),
                    'weight': region_data.get('weight', 1.0),
                    'color': COLOR_MAPPING.get(region_data.get('color', 'blue'), (255, 0, 0))
                }

        # Override with saved regions from regions_config.json
        if os.path.exists(REGIONS_FILE):
            try:
                with open(REGIONS_FILE, 'r') as f:
                    saved_regions = json.load(f)
                for cam_id, cam_regions in saved_regions.items():
                    if cam_id not in regions:
                        regions[cam_id] = {}
                    for region_name, region_data in cam_regions.items():
                        if 'vertices' in region_data:
                            regions[cam_id][region_name] = {
                                'vertices': np.array(region_data['vertices'], dtype=np.int32),
                                'weight': region_data.get('weight', 1.0),
                                'color': COLOR_MAPPING.get(region_data.get('color', 'blue'), (255, 0, 0))
                            }
                logger.debug("[CONFIG] Loaded saved regions from %s", REGIONS_FILE)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("[CONFIG] Could not load saved regions: %s", e)

        logger.debug("[CONFIG] Regions loaded for cameras: %s", list(regions.keys()))
        return regions

    def save_regions(self, cam_id, region_list):
        """Save region configuration for a camera."""
        logger.debug("[CONFIG] Saving regions for %s", cam_id)
        saved = {}
        if os.path.exists(REGIONS_FILE):
            try:
                with open(REGIONS_FILE, 'r') as f:
                    saved = json.load(f)
            except (json.JSONDecodeError, IOError):
                saved = {}

        saved[cam_id] = {}
        for region in region_list:
            vertices = region.get('vertices', [])
            saved[cam_id][region['type']] = {
                'vertices': [list(v) if isinstance(v, (list, tuple)) else v for v in vertices],
                'weight': region.get('weight', 1.0),
                'color': region.get('color', 'blue')
            }

        with open(REGIONS_FILE, 'w') as f:
            json.dump(saved, f, indent=4)
        logger.info("[CONFIG] Regions saved for %s", cam_id)

    # --- Directory initialization ---

    def init_directories(self):
        """Create necessary directories based on configured cameras."""
        logger.debug("[CONFIG] Initializing directories")
        os.makedirs('static/temp/', exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        for cam_id in self.all_cameras:
            cam_dir = os.path.join(self.processed_folder, cam_id)
            os.makedirs(cam_dir, exist_ok=True)
            logger.debug("[CONFIG] Directory ensured: %s", cam_dir)

    # --- GPU initialization ---

    def configure_gpu(self):
        """Configure GPU settings for optimal performance on Jetson Orin."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                logger.info("[GPU] Device: %s", gpu_name)
                logger.info("[GPU] Memory: %.1f GB", gpu_mem)
                logger.info("[GPU] cuDNN benchmark mode: enabled")

                # Jetson Orin specific optimizations
                if 'orin' in gpu_name.lower() or 'jetson' in gpu_name.lower():
                    logger.info("[GPU] Jetson Orin detected - applying optimizations")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info("[GPU] TF32 mode enabled for Jetson optimization")
            else:
                logger.info("[GPU] No CUDA GPU available, running on CPU")
        except ImportError:
            logger.warning("[GPU] PyTorch not installed, GPU features disabled")


# Global config instance
config = AppConfig()


def initialize_environment():
    """Initialize the application environment."""
    config.init_directories()
    config.configure_gpu()
    regions = config.load_regions()
    logger.info("[INIT] Environment initialized with %d cameras", config.num_cameras)
    return regions

