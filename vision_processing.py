"""
GPU-optimized multi-camera video processing module for Zebra Crossing
vehicle detection. Designed for NVIDIA Jetson Orin deployment.

Cyclic processing: each enabled camera is processed for config.process_duration
seconds, then the loop moves to the next camera.
"""

import os
import time
import logging

import cv2
import numpy as np
from ultralytics import YOLO

from config import config, initialize_environment, COLOR_MAPPING
import database

logger = logging.getLogger(__name__)

# Proximity threshold (pixels) for zebra-line crossing detection
CROSSING_THRESHOLD = 30

# ---------------------------------------------------------------------------
# Module-level initialisation
# ---------------------------------------------------------------------------

regions = initialize_environment()
USE_SAHI = config.use_sahi


def load_model():
    """Load YOLOv8 nano model and move it to the configured device."""
    logger.info("[VISION] Loading YOLOv8n model (device=%s)", config.device)
    mdl = YOLO('yolov8n.pt')
    mdl.to(config.device)
    if config.device == 'cuda':
        mdl.model.half()  # FP16 for Jetson performance
        logger.info("[VISION] FP16 half-precision enabled for CUDA")
    logger.info("[VISION] Model loaded successfully on %s", config.device)
    return mdl


model = load_model()

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def detect_vehicles(frame, cam_id):
    """Run YOLOv8 inference on *frame* and return filtered detections.

    Returns a list of dicts:
        [{'class': 'car', 'confidence': 0.85, 'bbox': [x1, y1, x2, y2]}, ...]
    """
    t0 = time.time()
    results = model(frame, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in config.vehicle_classes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                })
    elapsed = time.time() - t0
    logger.debug("[VISION] %s: %d vehicles detected in %.3fs",
                 cam_id, len(detections), elapsed)
    return detections


def check_zebra_crossing(detections, cam_regions, cam_id):
    """Check if detected vehicles cross the zebra line.

    A vehicle crosses if its bottom-centre y-coordinate is within
    CROSSING_THRESHOLD pixels of the zebra line.

    Returns:
        crossing_count (int), vehicle_type_counts (dict)
    """
    crossing_count = 0
    vehicle_type_counts = {}

    zebra = cam_regions.get('Zebra')
    if zebra is None:
        return crossing_count, vehicle_type_counts

    vertices = zebra['vertices']
    if len(vertices) < 2:
        return crossing_count, vehicle_type_counts

    # Zebra line y-value: average of endpoint y-coordinates
    line_y = float(np.mean(vertices[:, 1]))

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        bottom_centre_y = y2
        if abs(bottom_centre_y - line_y) <= CROSSING_THRESHOLD:
            crossing_count += 1
            cls = det['class']
            vehicle_type_counts[cls] = vehicle_type_counts.get(cls, 0) + 1

    if crossing_count:
        logger.debug("[VISION] %s: %d vehicles crossing zebra line",
                     cam_id, crossing_count)
    return crossing_count, vehicle_type_counts


def annotate_frame(frame, detections, cam_regions, cam_id):
    """Draw bounding boxes, labels, and zebra line on *frame* (in-place)."""
    # Draw zebra crossing line
    zebra = cam_regions.get('Zebra')
    if zebra is not None:
        verts = zebra['vertices']
        if len(verts) >= 2:
            pt1 = tuple(verts[0])
            pt2 = tuple(verts[-1])
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # blue in BGR

    # Draw vehicle bounding boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Camera ID overlay
    cv2.putText(frame, cam_id, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


# ---------------------------------------------------------------------------
# Per-camera processing cycle
# ---------------------------------------------------------------------------


def process_camera_cycle(cam_id, rtsp_url, duration):
    """Open *rtsp_url*, process frames for *duration* seconds, then release."""
    global regions

    logger.info("[VISION] Starting cycle for %s (duration=%ds)", cam_id, duration)

    cam_regions = regions.get(cam_id, {})
    save_dir = os.path.join(config.processed_folder, cam_id)
    os.makedirs(save_dir, exist_ok=True)

    # Open RTSP stream with TCP transport
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        logger.warning("[VISION] %s: cannot open stream %s", cam_id, rtsp_url)
        return

    frame_count = 0
    total_vehicle_count = 0
    total_crossing_count = 0
    aggregated_types = {}
    cycle_start = time.time()
    last_save_time = cycle_start
    processing_time_sum = 0.0

    try:
        while time.time() - cycle_start < duration:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[VISION] %s: frame read failed, ending cycle",
                               cam_id)
                break

            frame_count += 1
            if frame_count % config.frame_skip != 0:
                continue

            t0 = time.time()

            detections = detect_vehicles(frame, cam_id)
            crossing_count, type_counts = check_zebra_crossing(
                detections, cam_regions, cam_id)

            total_vehicle_count += len(detections)
            total_crossing_count += crossing_count
            for k, v in type_counts.items():
                aggregated_types[k] = aggregated_types.get(k, 0) + v

            annotate_frame(frame, detections, cam_regions, cam_id)

            # Save annotated frame at configured interval
            now = time.time()
            if now - last_save_time >= config.image_save_interval:
                fname = f"{cam_id}_{int(now)}.jpg"
                fpath = os.path.join(save_dir, fname)
                cv2.imwrite(fpath, frame)
                logger.debug("[VISION] %s: saved %s", cam_id, fname)
                last_save_time = now

            processing_time_sum += time.time() - t0

    except Exception:
        logger.exception("[VISION] %s: error during processing cycle", cam_id)
    finally:
        cap.release()

    # Insert summary detection data into database
    elapsed = time.time() - cycle_start
    density = total_vehicle_count / max(elapsed, 0.001)
    try:
        database.insert_detection(
            cam_id=cam_id,
            vehicle_count=total_vehicle_count,
            density=round(density, 4),
            weighted_count=total_vehicle_count,
            weighted_density=round(density, 4),
            vdc=total_crossing_count,
            processing_time=round(processing_time_sum, 4),
            vehicle_types=aggregated_types,
        )
    except Exception:
        logger.exception("[VISION] %s: failed to insert detection data", cam_id)

    logger.info(
        "[VISION] %s cycle done: frames=%d vehicles=%d crossings=%d "
        "elapsed=%.1fs",
        cam_id, frame_count, total_vehicle_count, total_crossing_count,
        elapsed,
    )


# ---------------------------------------------------------------------------
# Main cyclic processing loop
# ---------------------------------------------------------------------------


def cyclic_processing():
    """Continuously cycle through enabled cameras, processing each in turn."""
    global regions

    logger.info("[VISION] Cyclic processing started")

    while True:
        cameras = config.cameras  # dynamic – picks up changes at runtime
        if not cameras:
            logger.warning("[VISION] No enabled cameras, sleeping 5s")
            time.sleep(5)
            continue

        # Refresh regions each full cycle
        regions = config.load_regions()

        for cam_id, cam_info in cameras.items():
            rtsp_url = cam_info.get('rtsp_url', '')
            if not rtsp_url:
                logger.warning("[VISION] %s: no RTSP URL configured, skipping",
                               cam_id)
                continue
            try:
                process_camera_cycle(
                    cam_id, rtsp_url, duration=config.process_duration)
            except Exception:
                logger.exception(
                    "[VISION] %s: unhandled error, continuing to next camera",
                    cam_id)
