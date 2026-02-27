"""
Zebra Crossing Vehicle Analytics v2 - Main Entry Point.
Initializes logging, database, background processing, and starts the Flask server.
Designed for NVIDIA Jetson Orin with GPU acceleration.
"""

import logging
import sys
from threading import Thread

from config import config


def setup_logging():
    """Configure application-wide logging based on config settings."""
    log_level = getattr(logging, config.log_level, logging.DEBUG)
    log_format = config.log_format

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("[APP] Logging configured at level %s", config.log_level)
    return logger


def main():
    """Application entry point."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("[APP] Zebra Crossing Vehicle Analytics v2")
    logger.info("[APP] Starting up...")
    logger.info("=" * 60)

    # Log system info
    logger.info("[APP] Configured cameras: %d", config.num_cameras)
    for cam_id, cam_info in config.cameras.items():
        logger.info("[APP]   %s: %s (RTSP: %s)",
                     cam_id, cam_info.get('name', cam_id),
                     cam_info.get('rtsp_url', 'N/A'))
    logger.info("[APP] Compute device: %s", config.device)
    logger.info("[APP] Server: %s:%d (debug=%s)",
                 config.host, config.port, config.debug)

    # Initialize database
    import database
    database.init_db()
    logger.info("[APP] Database initialized")

    # Start background vision processing
    import vision_processing
    processing_thread = Thread(
        target=vision_processing.cyclic_processing,
        daemon=True,
        name="VisionProcessingThread"
    )
    processing_thread.start()
    logger.info("[APP] Background processing thread started")

    # Start Flask application
    from routes import app
    logger.info("[APP] Starting Flask server on %s:%d", config.host, config.port)
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
        threaded=True,
        use_reloader=False  # Avoid double-starting processing thread
    )


if __name__ == "__main__":
    main()
