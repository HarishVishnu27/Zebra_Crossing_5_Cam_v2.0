"""
Database module for Zebra Crossing Vehicle Analytics v2.
SQLite-based storage with thread-safe operations and IST timestamps.
Supports dynamic camera count from config.
"""

import os
import io
import csv
import json
import sqlite3
import logging
import threading
from datetime import datetime, timedelta, timezone

from config import config

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

_db_lock = threading.Lock()


def _now_ist():
    """Return current datetime in IST."""
    return datetime.now(IST)


def _today_midnight_ist():
    """Return today's midnight in IST as ISO string."""
    now = _now_ist()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.strftime('%Y-%m-%d %H:%M:%S')


def _get_conn():
    """Create a new SQLite connection with WAL mode."""
    db_path = config.database_path
    logger.debug("[DB] Opening connection to %s", db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the traffic_data table and indexes if they do not exist."""
    logger.info("[DB] Initializing database at %s", config.database_path)
    with _db_lock:
        conn = _get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traffic_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cam_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    vehicle_count INTEGER DEFAULT 0,
                    density REAL DEFAULT 0.0,
                    weighted_count INTEGER DEFAULT 0,
                    weighted_density REAL DEFAULT 0.0,
                    vdc INTEGER DEFAULT 0,
                    processing_time REAL DEFAULT 0.0,
                    vehicle_types TEXT DEFAULT '{}'
                )
            """)
            logger.debug("[DB] Table traffic_data ensured")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cam_id
                ON traffic_data (cam_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON traffic_data (timestamp)
            """)
            logger.debug("[DB] Indexes on cam_id and timestamp ensured")
            conn.commit()
            logger.info("[DB] Database initialization complete")
        except Exception:
            logger.exception("[DB] Error initializing database")
            raise
        finally:
            conn.close()


def insert_detection(cam_id, vehicle_count, density, weighted_count,
                     weighted_density, vdc, processing_time,
                     vehicle_types=None):
    """Insert a detection record with the current IST timestamp."""
    ts = _now_ist().strftime('%Y-%m-%d %H:%M:%S')
    vt_json = json.dumps(vehicle_types if vehicle_types else {})
    logger.debug(
        "[DB] Inserting detection: cam=%s count=%d density=%.2f "
        "weighted_count=%d weighted_density=%.2f vdc=%d proc=%.3fs types=%s",
        cam_id, vehicle_count, density, weighted_count, weighted_density,
        vdc, processing_time, vt_json
    )
    with _db_lock:
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO traffic_data
                   (cam_id, timestamp, vehicle_count, density,
                    weighted_count, weighted_density, vdc,
                    processing_time, vehicle_types)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cam_id, ts, vehicle_count, density, weighted_count,
                 weighted_density, vdc, processing_time, vt_json)
            )
            conn.commit()
            logger.debug("[DB] Detection inserted for %s at %s", cam_id, ts)
        except Exception:
            logger.exception("[DB] Error inserting detection for %s", cam_id)
            raise
        finally:
            conn.close()


def get_camera_stats_today(cam_id):
    """Get today's stats for a camera since midnight IST."""
    midnight = _today_midnight_ist()
    logger.debug("[DB] Getting today's stats for %s since %s", cam_id, midnight)
    stats = {
        'total_count': 0,
        'latest_detection': None,
        'vehicle_counts': {}
    }
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                """SELECT COALESCE(SUM(vehicle_count), 0) AS total,
                          MAX(timestamp) AS latest
                   FROM traffic_data
                   WHERE cam_id = ? AND timestamp >= ?""",
                (cam_id, midnight)
            ).fetchone()
            if row:
                stats['total_count'] = row['total']
                stats['latest_detection'] = row['latest']
                logger.debug("[DB] %s today total=%d latest=%s",
                             cam_id, row['total'], row['latest'])

            rows = conn.execute(
                """SELECT vehicle_types FROM traffic_data
                   WHERE cam_id = ? AND timestamp >= ?""",
                (cam_id, midnight)
            ).fetchall()
            aggregated = {}
            for r in rows:
                try:
                    vt = json.loads(r['vehicle_types']) if r['vehicle_types'] else {}
                except (json.JSONDecodeError, TypeError):
                    vt = {}
                for k, v in vt.items():
                    aggregated[k] = aggregated.get(k, 0) + v
            stats['vehicle_counts'] = aggregated
            logger.debug("[DB] %s vehicle_counts=%s", cam_id, aggregated)
        except Exception:
            logger.exception("[DB] Error getting stats for %s", cam_id)
        finally:
            conn.close()
    return stats


def get_last_processed(cam_id):
    """Get the most recent processed image filename for a camera."""
    cam_dir = os.path.join(config.processed_folder, cam_id)
    logger.debug("[DB] Looking for last processed image in %s", cam_dir)
    if not os.path.isdir(cam_dir):
        logger.debug("[DB] Directory does not exist: %s", cam_dir)
        return None
    files = [f for f in os.listdir(cam_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        logger.debug("[DB] No processed images found for %s", cam_id)
        return None
    files.sort(reverse=True)
    logger.debug("[DB] Last processed for %s: %s", cam_id, files[0])
    return files[0]


def _resolve_date_range(preset, start_date=None, end_date=None):
    """Resolve a preset or custom date range to (start, end) ISO strings."""
    now = _now_ist()
    if preset == 'today':
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif preset == 'yesterday':
        yesterday = now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
    elif preset == 'last24':
        start = now - timedelta(hours=24)
        end = now
    elif preset == 'lastweek':
        start = now - timedelta(days=7)
        end = now
    elif preset == 'lastmonth':
        start = now - timedelta(days=30)
        end = now
    elif preset == 'custom' and start_date and end_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=IST)
        end = datetime.strptime(end_date, '%Y-%m-%d').replace(
            hour=23, minute=59, second=59, tzinfo=IST)
    else:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
    s = start.strftime('%Y-%m-%d %H:%M:%S')
    e = end.strftime('%Y-%m-%d %H:%M:%S')
    logger.debug("[DB] Date range resolved: %s -> %s (preset=%s)", s, e, preset)
    return s, e


def get_analytics_data(cam_id=None, preset='today', start_date=None,
                       end_date=None, vehicle_type=None):
    """Flexible analytics query with preset date ranges.

    If cam_id is 'consolidated', aggregate across all cameras.
    """
    start, end = _resolve_date_range(preset, start_date, end_date)
    consolidated = (cam_id == 'consolidated')
    logger.debug(
        "[DB] Analytics query: cam=%s preset=%s range=%s..%s vehicle_type=%s",
        cam_id, preset, start, end, vehicle_type
    )
    result = {
        'total_count': 0,
        'vehicle_counts': {},
        'since': start,
        'data': []
    }
    with _db_lock:
        conn = _get_conn()
        try:
            if consolidated or cam_id is None:
                query = """SELECT * FROM traffic_data
                           WHERE timestamp >= ? AND timestamp <= ?
                           ORDER BY timestamp DESC"""
                params = (start, end)
            else:
                query = """SELECT * FROM traffic_data
                           WHERE cam_id = ? AND timestamp >= ? AND timestamp <= ?
                           ORDER BY timestamp DESC"""
                params = (cam_id, start, end)
            rows = conn.execute(query, params).fetchall()

            total = 0
            aggregated_vt = {}
            data_rows = []
            for r in rows:
                try:
                    vt = json.loads(r['vehicle_types']) if r['vehicle_types'] else {}
                except (json.JSONDecodeError, TypeError):
                    vt = {}

                # Filter by vehicle_type if specified
                if vehicle_type:
                    if vehicle_type not in vt:
                        continue

                total += r['vehicle_count']
                for k, v in vt.items():
                    aggregated_vt[k] = aggregated_vt.get(k, 0) + v
                data_rows.append(dict(r))

            result['total_count'] = total
            result['vehicle_counts'] = aggregated_vt
            result['data'] = data_rows
            logger.debug(
                "[DB] Analytics result: total=%d rows=%d vehicle_counts=%s",
                total, len(data_rows), aggregated_vt
            )
        except Exception:
            logger.exception("[DB] Error in analytics query")
        finally:
            conn.close()
    return result


def get_analytics_page_data(start_date=None, end_date=None):
    """Return per-camera data and aggregates for the analytics page."""
    if start_date and end_date:
        preset = 'custom'
    else:
        preset = 'today'
    start, end = _resolve_date_range(preset, start_date, end_date)
    logger.debug("[DB] Analytics page data: %s -> %s", start, end)

    camera_ids = config.camera_ids
    logger.debug("[DB] Dynamic camera list: %s", camera_ids)

    page_data = {
        'cameras': {},
        'avg_density': 0.0,
        'peak_count': 0
    }

    total_density = 0.0
    density_count = 0
    peak = 0

    with _db_lock:
        conn = _get_conn()
        try:
            for cid in camera_ids:
                rows = conn.execute(
                    """SELECT * FROM traffic_data
                       WHERE cam_id = ? AND timestamp >= ? AND timestamp <= ?
                       ORDER BY timestamp DESC""",
                    (cid, start, end)
                ).fetchall()

                cam_total = 0
                cam_vt = {}
                cam_data = []
                for r in rows:
                    cam_total += r['vehicle_count']
                    total_density += r['density']
                    density_count += 1
                    if r['vehicle_count'] > peak:
                        peak = r['vehicle_count']
                    try:
                        vt = json.loads(r['vehicle_types']) if r['vehicle_types'] else {}
                    except (json.JSONDecodeError, TypeError):
                        vt = {}
                    for k, v in vt.items():
                        cam_vt[k] = cam_vt.get(k, 0) + v
                    cam_data.append(dict(r))

                page_data['cameras'][cid] = {
                    'total_count': cam_total,
                    'vehicle_counts': cam_vt,
                    'data': cam_data
                }
                logger.debug("[DB] Page data for %s: total=%d", cid, cam_total)

            page_data['avg_density'] = (
                round(total_density / density_count, 4) if density_count else 0.0
            )
            page_data['peak_count'] = peak
            logger.debug(
                "[DB] Page aggregates: avg_density=%.4f peak_count=%d",
                page_data['avg_density'], peak
            )
        except Exception:
            logger.exception("[DB] Error getting analytics page data")
        finally:
            conn.close()
    return page_data


def export_csv(cam_id=None, preset='today', start_date=None, end_date=None,
               vehicle_type=None):
    """Export analytics data as a CSV string."""
    logger.debug(
        "[DB] CSV export: cam=%s preset=%s start=%s end=%s type=%s",
        cam_id, preset, start_date, end_date, vehicle_type
    )
    data = get_analytics_data(
        cam_id=cam_id, preset=preset, start_date=start_date,
        end_date=end_date, vehicle_type=vehicle_type
    )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'id', 'cam_id', 'timestamp', 'vehicle_count', 'density',
        'weighted_count', 'weighted_density', 'vdc',
        'processing_time', 'vehicle_types'
    ])
    for row in data['data']:
        writer.writerow([
            row.get('id', ''),
            row.get('cam_id', ''),
            row.get('timestamp', ''),
            row.get('vehicle_count', 0),
            row.get('density', 0.0),
            row.get('weighted_count', 0),
            row.get('weighted_density', 0.0),
            row.get('vdc', 0),
            row.get('processing_time', 0.0),
            row.get('vehicle_types', '{}')
        ])
    csv_str = output.getvalue()
    logger.debug("[DB] CSV export complete: %d data rows", len(data['data']))
    return csv_str
