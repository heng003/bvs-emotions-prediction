import logging
import uuid
from datetime import datetime, timezone, timedelta

# Define Asia/Kuala_Lumpur Timezone (UTC+8)
KL_TZ = timezone(timedelta(hours=8))
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel
from database import supabase_client
from fitbit_client import FitbitClient
from emotion_model import EmotionPredictor
from status_tracker import status_tracker, router as status_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Emotion Predictor (Loads trained model or uses heuristic fallback)
predictor = EmotionPredictor()

class PredictionRequest(BaseModel):
    hr: float
    hrv: float
    azm: float

async def process_user_device(user: dict):
    """
    Core logic for a single user:
    1. Get Device Creds
    2. Fetch Fitbit Data (5 min window)
    3. Predict Emotion
    4. Save to Supabase
    """
    user_id = user.get('id')
    fitbit_api_success = False
    db_write_success = False
    error_msg = None
    processing_timestamp = datetime.now(timezone.utc)
    
    try:
        device_id = user['device_id']
        
        # 1. Fetch Device details
        device_response = supabase_client.table('devices').select('*').eq('id', device_id).execute()
        if not device_response.data:
            logger.warning(f"Device not found for user {user_id}")
            error_msg = "Device not found"
            status_tracker.log_user_processing(
                user_id=user_id,
                timestamp=processing_timestamp,
                emotion="unknown",
                confidence=0.0,
                db_write_success=False,
                fitbit_api_success=False,
                error=error_msg
            )
            return
        
        # Log DB Extraction Success
        logger.info(f"DB: Successfully extracted device details for user {user_id}")
        
        device = device_response.data[0]
        
        # Check if device is active
        if device.get('status') != 'active':
            logger.info(f"Device {device_id} for user {user_id} is not active. Status: {device.get('status')}")
            error_msg = f"Device not active: {device.get('status')}"
            status_tracker.log_user_processing(
                user_id=user_id,
                timestamp=processing_timestamp,
                emotion="unknown",
                confidence=0.0,
                db_write_success=False,
                fitbit_api_success=False,
                error=error_msg
            )
            return
        
        # 2. Initialize Fitbit Client (handles token refresh automatically)
        client = FitbitClient(device, user_id)
        
        # 3. Fetch Data (Last 5 minutes)
        # Returns a dict with averages or max outliers: {'hr': 75, 'hrv': 40, 'azm': 2}
        sensor_data = await client.fetch_last_5_min_data()
        
        if not sensor_data:
            logger.info(f"No recent data synced for user {user_id}")
            error_msg = "No recent Fitbit data"
            status_tracker.log_user_processing(
                user_id=user_id,
                timestamp=processing_timestamp,
                emotion="unknown",
                confidence=0.0,
                db_write_success=False,
                fitbit_api_success=False,
                error=error_msg
            )
            return

        # VALIDATION FIX: Ensure sensor_data is a dictionary (type check before .get())
        if not isinstance(sensor_data, dict):
            logger.error(f"FitbitClient returned invalid data type: {type(sensor_data)}. Expected dict. Data: {sensor_data}")
            error_msg = "Invalid Fitbit data format"
            status_tracker.log_user_processing(
                user_id=user_id,
                timestamp=processing_timestamp,
                emotion="unknown",
                confidence=0.0,
                db_write_success=False,
                fitbit_api_success=False,
                error=error_msg
            )
            return

        # Log Fitbit Extraction Success
        logger.info(f"Fitbit: Successfully extracted data for user {user_id}: {sensor_data}")
        fitbit_api_success = True

        # 4. Predict Emotion
        # Input format: [HeartRate, HRV, ActiveZoneMinutes]
        features = [
            sensor_data.get('hr', 0),
            sensor_data.get('hrv', 0),
            sensor_data.get('azm', 0)
        ]
        
        prediction = predictor.predict(features)
        
        # 5. Save Result
        # Ensure timestamp is saved with Asia/Kuala_Lumpur timezone (UTC+8)
        try:
            # Parse the timestamp string
            ts_str = sensor_data.get('timestamp')
            if ts_str:
                ts_obj = datetime.fromisoformat(ts_str)
                # If naive (no timezone), assume it was generated in KL time (local execution)
                if ts_obj.tzinfo is None:
                    ts_obj = ts_obj.replace(tzinfo=KL_TZ)
                else:
                    # If it has a timezone, convert to KL
                    ts_obj = ts_obj.astimezone(KL_TZ)
                
                final_timestamp = ts_obj.isoformat()
            else:
                final_timestamp = datetime.now(timezone.utc).astimezone(KL_TZ).isoformat()
        except Exception as e:
            logger.warning(f"Timestamp parsing failed ({e}), using current KL time.")
            # Fallback to current KL time if parsing fails
            final_timestamp = datetime.now(timezone.utc).astimezone(KL_TZ).isoformat()

        data_payload = {
            "user_id": user_id,
            "timestamp": final_timestamp,
            "predicted_emotion": prediction['emotion'],
            "emotion_confidence": prediction['confidence']
        }
        
        supabase_client.table('bvs_emotion').insert(data_payload).execute()
        logger.info(f"Saved prediction for {user_id}: {prediction['emotion']}")
        db_write_success = True
        
        # Log successful processing
        status_tracker.log_user_processing(
            user_id=user_id,
            timestamp=processing_timestamp,
            emotion=prediction['emotion'],
            confidence=prediction['confidence'],
            db_write_success=True,
            fitbit_api_success=True,
            error=None
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing user {user_id}: {error_msg}")
        # Log failed processing
        status_tracker.log_user_processing(
            user_id=user_id or "unknown",
            timestamp=processing_timestamp,
            emotion="unknown",
            confidence=0.0,
            db_write_success=False,
            fitbit_api_success=fitbit_api_success,
            error=error_msg
        )

async def scheduled_job():
    """
    Runs every 5 minutes.
    Fetches all active users with devices and processes them.
    """
    job_id = str(uuid.uuid4())
    job_start_time = datetime.now(timezone.utc)
    users_found = 0
    users_processed = 0
    users_failed = 0
    
    logger.info("Starting scheduled data fetch...")
    
    # Log job start
    status_tracker.log_scheduled_job_run(
        job_id=job_id,
        started_at=job_start_time,
        users_found=0,
        users_processed=0,
        users_failed=0,
        status="running"
    )
    
    try:
        # Fetch active users who have a device_id
        # FIX: Use .not_.is_('device_id', 'null') instead of .neq('device_id', 'null') for UUID columns
        response = supabase_client.table('users').select('*').not_.is_('device_id', 'null').execute()
        users = response.data
        
        users_found = len(users) if users else 0
        
        # Log DB User Extraction
        logger.info(f"DB: Extracted {users_found} active users.")
        
        # Update job with user count
        status_tracker.log_scheduled_job_run(
            job_id=job_id,
            started_at=job_start_time,
            users_found=users_found,
            users_processed=0,
            users_failed=0,
            status="running"
        )

        if not users:
            logger.info("No active users with devices found.")
            # Log completed job with no users
            status_tracker.log_scheduled_job_run(
                job_id=job_id,
                started_at=job_start_time,
                users_found=0,
                users_processed=0,
                users_failed=0,
                completed_at=datetime.now(timezone.utc),
                status="completed"
            )
            return

        # In production, you might want to use a task queue (Celery) here for parallelism
        for user in users:
            try:
                await process_user_device(user)
                users_processed += 1
            except Exception as e:
                logger.error(f"Failed to process user {user.get('id')}: {e}")
                users_failed += 1
        
        # Log job completion
        job_end_time = datetime.now(timezone.utc)
        status_tracker.log_scheduled_job_run(
            job_id=job_id,
            started_at=job_start_time,
            users_found=users_found,
            users_processed=users_processed,
            users_failed=users_failed,
            completed_at=job_end_time,
            status="completed"
        )
        logger.info(f"Scheduled job completed: {users_processed} processed, {users_failed} failed out of {users_found} users")
            
    except Exception as e:
        logger.error(f"DB Error during user fetch: {e}")
        # Log job failure
        status_tracker.log_scheduled_job_run(
            job_id=job_id,
            started_at=job_start_time,
            users_found=users_found,
            users_processed=users_processed,
            users_failed=users_failed,
            completed_at=datetime.now(timezone.utc),
            status="failed"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Test DB Connection on Startup
    try:
        # Simple query to check connectivity
        supabase_client.table('users').select("id").limit(1).execute()
        logger.info("Database connected successfully.")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

    # 2. Start Scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_job, 'interval', minutes=5)
    scheduler.start()
    yield
    # Shutdown logic if needed

app = FastAPI(lifespan=lifespan)

# Include status tracking routes
app.include_router(status_router)

@app.get("/")
async def root():
    return {"status": "System is running", "scheduler": "Active (5 min interval)"}

@app.post("/trigger-now")
async def trigger_manual():
    """Manually trigger the job for testing"""
    await scheduled_job()
    return {"message": "Job triggered"}

@app.post("/predict")
async def manual_predict(request: PredictionRequest):
    """
    Manually input values to test the emotion prediction model.
    Input:
    - hr: Heart Rate (bpm)
    - hrv: Heart Rate Variability (ms, RMSSD)
    - azm: Active Zone Minutes (0 or 1, or minutes count)
    """
    request_timestamp = datetime.now(timezone.utc)
    features = [request.hr, request.hrv, request.azm]
    prediction = predictor.predict(features)
    
    # Log manual prediction request
    status_tracker.log_manual_predict(
        timestamp=request_timestamp,
        hr=request.hr,
        hrv=request.hrv,
        azm=request.azm,
        emotion=prediction['emotion'],
        confidence=prediction['confidence']
    )
    
    return {
        "input": {
            "hr": request.hr,
            "hrv": request.hrv,
            "azm": request.azm
        },
        "prediction": prediction
    }