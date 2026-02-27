"""
Flask routes module for Zebra Crossing Vehicle Analytics v2.
Provides web UI and REST API endpoints with session-based auth.
"""

import os
import re
import json
import logging
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, Response,
)
from config import config
import database
import vision_processing

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.urandom(24)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


_CAM_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


def _validate_cam_id(cam_id):
    """Validate cam_id to prevent path traversal attacks."""
    if not cam_id or not _CAM_ID_RE.match(cam_id):
        return False
    return True


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    logger.debug("[ROUTES] /login %s", request.method)
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == config.username and password == config.password:
            session['logged_in'] = True
            logger.debug("[ROUTES] Login successful for %s", username)
            return redirect(url_for('index'))
        logger.debug("[ROUTES] Login failed for %s", username)
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')


@app.route('/logout')
def logout():
    logger.debug("[ROUTES] /logout")
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    logger.debug("[ROUTES] / (dashboard)")
    camera_ids = config.camera_ids
    latest_images = {}
    camera_stats = {}
    for cam_id in camera_ids:
        latest_images[cam_id] = database.get_last_processed(cam_id)
        camera_stats[cam_id] = database.get_camera_stats_today(cam_id)
    return render_template(
        'index.html',
        cameras=config.cameras,
        camera_ids=camera_ids,
        latest_images=latest_images,
        camera_stats=camera_stats,
    )


@app.route('/admin')
@login_required
def admin_panel():
    logger.debug("[ROUTES] /admin")
    return render_template(
        'admin.html',
        cameras=config.cameras,
        camera_ids=config.camera_ids,
    )


@app.route('/analytics')
@login_required
def zebra_crossing_analytics():
    logger.debug("[ROUTES] /analytics")
    return render_template(
        'analytics.html',
        cameras=config.cameras,
        camera_ids=config.camera_ids,
    )


# ---------------------------------------------------------------------------
# Camera / image API
# ---------------------------------------------------------------------------

@app.route('/get_last_processed/<cam_id>')
@login_required
def get_last_processed(cam_id):
    if not _validate_cam_id(cam_id):
        return jsonify({'status': 'error', 'message': 'Invalid camera ID'}), 400
    logger.debug("[ROUTES] /get_last_processed/%s", cam_id)
    filename = database.get_last_processed(cam_id)
    if filename:
        image_url = f'/static/processed/{cam_id}/{filename}'
        return jsonify({'status': 'success', 'image_url': image_url})
    return jsonify({'status': 'error', 'message': 'No processed image found'})


@app.route('/get_frame/<cam_id>')
@login_required
def get_frame(cam_id):
    if not _validate_cam_id(cam_id):
        return jsonify({'status': 'error', 'message': 'Invalid camera ID'}), 400
    logger.debug("[ROUTES] /get_frame/%s", cam_id)
    try:
        import cv2
        cameras = config.cameras
        if cam_id not in cameras:
            return jsonify({'status': 'error', 'message': 'Camera not found'})
        rtsp_url = cameras[cam_id]['rtsp_url']
        cap = cv2.VideoCapture(rtsp_url)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({'status': 'error', 'message': 'Failed to capture frame'})
        os.makedirs('static/temp', exist_ok=True)
        frame_path = f'static/temp/{cam_id}_frame.jpg'
        cv2.imwrite(frame_path, frame)
        frame_url = f'/static/temp/{cam_id}_frame.jpg'
        logger.debug("[ROUTES] Frame saved: %s", frame_path)
        return jsonify({'status': 'success', 'frame_url': frame_url})
    except Exception as e:
        logger.exception("[ROUTES] Error capturing frame for %s", cam_id)
        return jsonify({'status': 'error', 'message': str(e)})


# ---------------------------------------------------------------------------
# Region configuration API
# ---------------------------------------------------------------------------

@app.route('/get_regions')
@login_required
def get_regions():
    logger.debug("[ROUTES] /get_regions")
    try:
        regions = config.load_regions()
        serializable = {}
        for cam_id, cam_regions in regions.items():
            serializable[cam_id] = {}
            for name, data in cam_regions.items():
                vertices = data.get('vertices', [])
                if hasattr(vertices, 'tolist'):
                    vertices = vertices.tolist()
                serializable[cam_id][name] = {
                    'vertices': vertices,
                    'weight': data.get('weight', 1.0),
                    'color': data.get('color', 'blue'),
                }
        return jsonify(serializable)
    except Exception as e:
        logger.exception("[ROUTES] Error loading regions")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/update_regions', methods=['POST'])
@login_required
def update_regions():
    logger.debug("[ROUTES] /update_regions")
    try:
        payload = request.get_json()
        cam_id = payload.get('cam_id')
        region_list = payload.get('regions', [])
        config.save_regions(cam_id, region_list)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.exception("[ROUTES] Error updating regions")
        return jsonify({'status': 'error', 'message': str(e)})


# ---------------------------------------------------------------------------
# Processing control API
# ---------------------------------------------------------------------------

@app.route('/toggle_sahi', methods=['POST'])
@login_required
def toggle_sahi():
    logger.debug("[ROUTES] /toggle_sahi")
    try:
        payload = request.get_json()
        use_sahi = payload.get('use_sahi', False)
        vision_processing.USE_SAHI = use_sahi
        state = 'enabled' if use_sahi else 'disabled'
        logger.debug("[ROUTES] SAHI %s", state)
        return jsonify({'status': 'success', 'message': f'SAHI {state}'})
    except Exception as e:
        logger.exception("[ROUTES] Error toggling SAHI")
        return jsonify({'status': 'error', 'message': str(e)})


# ---------------------------------------------------------------------------
# Analytics API
# ---------------------------------------------------------------------------

@app.route('/api/analytics_data')
@login_required
def analytics_data():
    logger.debug("[ROUTES] /api/analytics_data")
    cam_id = request.args.get('cam_id')
    preset = request.args.get('preset', 'today')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    vehicle_type = request.args.get('vehicle_type')
    stats = database.get_analytics_data(
        cam_id=cam_id,
        preset=preset,
        start_date=start_date,
        end_date=end_date,
        vehicle_type=vehicle_type,
    )
    return jsonify({'stats': stats})


@app.route('/download_zebra_data')
@login_required
def download_zebra_data():
    logger.debug("[ROUTES] /download_zebra_data")
    cam_id = request.args.get('cam_id')
    preset = request.args.get('preset', 'today')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    vehicle_type = request.args.get('vehicle_type')
    csv_str = database.export_csv(
        cam_id=cam_id,
        preset=preset,
        start_date=start_date,
        end_date=end_date,
        vehicle_type=vehicle_type,
    )
    return Response(
        csv_str,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=zebra_data.csv'},
    )


# ---------------------------------------------------------------------------
# System / config API
# ---------------------------------------------------------------------------

@app.route('/api/system_info')
@login_required
def system_info():
    logger.debug("[ROUTES] /api/system_info")
    info = {
        'cameras_count': config.num_cameras,
        'device': config.device,
        'gpu': {},
    }
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu'] = {
                'available': True,
                'name': torch.cuda.get_device_name(0),
                'memory_gb': round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
                ),
            }
        else:
            info['gpu'] = {'available': False}
    except ImportError:
        info['gpu'] = {'available': False, 'note': 'PyTorch not installed'}
    except Exception:
        logger.exception("[ROUTES] Error querying GPU info")
        info['gpu'] = {'available': False, 'note': 'Error querying GPU'}
    return jsonify(info)


@app.route('/api/config')
@login_required
def get_config():
    logger.debug("[ROUTES] /api/config")
    return jsonify({
        'cameras': config.all_cameras,
        'processing': {
            'frame_skip': config.frame_skip,
            'vehicle_threshold': config.vehicle_threshold,
            'process_duration': config.process_duration,
            'image_save_interval': config.image_save_interval,
            'use_sahi': config.use_sahi,
            'vehicle_classes': config.vehicle_classes,
            'device': config.device,
        },
    })


@app.route('/api/config/camera', methods=['POST'])
@login_required
def upsert_camera():
    logger.debug("[ROUTES] /api/config/camera POST")
    try:
        payload = request.get_json()
        cam_id = payload.get('cam_id')
        name = payload.get('name', cam_id)
        rtsp_url = payload.get('rtsp_url', '')
        enabled = payload.get('enabled', True)

        if cam_id in config.all_cameras:
            config.update_camera(cam_id, name=name, rtsp_url=rtsp_url, enabled=enabled)
        else:
            config.add_camera(cam_id, name, rtsp_url, enabled)

        # Ensure processed directory exists for the camera
        cam_dir = os.path.join(config.processed_folder, cam_id)
        os.makedirs(cam_dir, exist_ok=True)
        logger.debug("[ROUTES] Camera upserted: %s", cam_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.exception("[ROUTES] Error upserting camera")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/config/camera/<cam_id>', methods=['DELETE'])
@login_required
def delete_camera(cam_id):
    if not _validate_cam_id(cam_id):
        return jsonify({'status': 'error', 'message': 'Invalid camera ID'}), 400
    logger.debug("[ROUTES] /api/config/camera/%s DELETE", cam_id)
    try:
        config.remove_camera(cam_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.exception("[ROUTES] Error removing camera %s", cam_id)
        return jsonify({'status': 'error', 'message': str(e)})
