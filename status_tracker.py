"""
BVS Service Status Tracker

Tracks scheduled job runs, user processing results, and manual requests for status monitoring.
Provides endpoints for health checks and service status.
"""

import logging
import datetime
import uuid
from collections import deque
from typing import Dict, List, Optional
from threading import Lock
from fastapi import APIRouter

logger = logging.getLogger(__name__)

# Initialize router for status endpoints
router = APIRouter()


class StatusTracker:
    """Track scheduled job runs, user processing results, and manual requests for status monitoring."""
    
    def __init__(
        self,
        max_job_runs: int = 50,
        max_user_results: int = 100,
        max_manual_requests: int = 50
    ):
        self.job_runs = deque(maxlen=max_job_runs)
        self.user_results = deque(maxlen=max_user_results)
        self.manual_requests = deque(maxlen=max_manual_requests)
        self.lock = Lock()
    
    def log_scheduled_job_run(
        self,
        job_id: str,
        started_at: datetime.datetime,
        users_found: int = 0,
        users_processed: int = 0,
        users_failed: int = 0,
        completed_at: Optional[datetime.datetime] = None,
        status: str = "running"
    ):
        """Log a scheduled job run."""
        with self.lock:
            job_entry = {
                "job_id": job_id,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat() if completed_at else None,
                "users_found": users_found,
                "users_processed": users_processed,
                "users_failed": users_failed,
                "status": status
            }
            # Update existing job if still running, otherwise append new
            if status == "running":
                # Check if there's an incomplete job
                for i, job in enumerate(self.job_runs):
                    if job.get("status") == "running" and job.get("job_id") == job_id:
                        self.job_runs[i] = job_entry
                        return
                # No incomplete job found, append new
                self.job_runs.append(job_entry)
            else:
                # Completed job - update if exists, otherwise append
                for i, job in enumerate(self.job_runs):
                    if job.get("job_id") == job_id:
                        self.job_runs[i] = job_entry
                        return
                self.job_runs.append(job_entry)
    
    def log_user_processing(
        self,
        user_id: str,
        timestamp: datetime.datetime,
        emotion: str,
        confidence: float,
        db_write_success: bool = False,
        fitbit_api_success: bool = True,
        error: Optional[str] = None
    ):
        """Log a user processing result."""
        with self.lock:
            self.user_results.append({
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "emotion": emotion,
                "emotion_confidence": confidence,
                "db_write_success": db_write_success,
                "fitbit_api_success": fitbit_api_success,
                "error": error,
                "status": "completed" if not error else "failed"
            })
    
    def log_manual_predict(
        self,
        timestamp: datetime.datetime,
        hr: float,
        hrv: float,
        azm: float,
        emotion: str,
        confidence: float
    ):
        """Log a manual prediction request."""
        with self.lock:
            self.manual_requests.append({
                "timestamp": timestamp.isoformat(),
                "input": {
                    "hr": hr,
                    "hrv": hrv,
                    "azm": azm
                },
                "prediction": {
                    "emotion": emotion,
                    "confidence": confidence
                },
                "status": "completed"
            })
    
    def get_recent_job_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent scheduled job runs (newest first)."""
        with self.lock:
            return list(reversed(list(self.job_runs)[-limit:]))
    
    def get_recent_user_results(self, limit: int = 20) -> List[Dict]:
        """Get recent user processing results (newest first)."""
        with self.lock:
            return list(reversed(list(self.user_results)[-limit:]))
    
    def get_recent_manual_requests(self, limit: int = 10) -> List[Dict]:
        """Get recent manual prediction requests (newest first)."""
        with self.lock:
            return list(reversed(list(self.manual_requests)[-limit:]))


# Global status tracker instance
status_tracker = StatusTracker()


@router.get("/health")
async def health():
    """Health check endpoint for Cloud Run."""
    try:
        return {
            "status": "healthy",
            "service": "bvs",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "bvs",
            "error": str(e),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }


@router.get("/bvs/status")
async def get_bvs_service_status():
    """
    Get detailed BVS service status for cloud dashboard monitoring.
    
    Returns real-time information about:
    - Recent scheduled job runs
    - Recent user processing results
    - Recent manual prediction requests
    """
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Get recent job runs
        recent_job_runs = status_tracker.get_recent_job_runs(limit=10)
        
        # Get recent user results
        recent_user_results = status_tracker.get_recent_user_results(limit=20)
        
        # Get recent manual requests
        recent_manual_requests = status_tracker.get_recent_manual_requests(limit=10)
        
        # Get last job run info
        last_job_run = recent_job_runs[0] if recent_job_runs else None
        
        # Get last successful user result
        last_successful_result = None
        for result in recent_user_results:
            if result.get("status") == "completed" and result.get("db_write_success"):
                last_successful_result = result
                break
        
        return {
            "service": "bvs",
            "timestamp": now.isoformat(),
            "status": "healthy",
            "last_job_run": last_job_run,
            "recent_job_runs": recent_job_runs[:5],  # Last 5 job runs
            "recent_user_results": recent_user_results[:10],  # Last 10 user results
            "recent_manual_requests": recent_manual_requests[:5],  # Last 5 manual requests
            "uptime": "unknown"  # Could be enhanced with actual uptime tracking
        }
        
    except Exception as e:
        logger.error(f"Error getting BVS service status: {e}", exc_info=True)
        return {
            "service": "bvs",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "error",
            "error": str(e),
            "last_job_run": None,
            "recent_job_runs": [],
            "recent_user_results": [],
            "recent_manual_requests": []
        }
