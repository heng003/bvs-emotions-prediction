import logging
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
    try:
        user_id = user['id']
        device_id = user['device_id']
        
        # 1. Fetch Device details
        device_response = supabase_client.table('devices').select('*').eq('id', device_id).execute()
        if not device_response.data:
            logger.warning(f"Device not found for user {user_id}")
            return
        
        # Log DB Extraction Success
        logger.info(f"DB: Successfully extracted device details for user {user_id}")
        
        device = device_response.data[0]
        
        # Check if device is active
        if device.get('status') != 'active':
            logger.info(f"Device {device_id} for user {user_id} is not active. Status: {device.get('status')}")
            return
        
        # 2. Initialize Fitbit Client (handles token refresh automatically)
        client = FitbitClient(device, user_id)
        
        # 3. Fetch Data (Last 5 minutes)
        # Returns a dict with averages or max outliers: {'hr': 75, 'hrv': 40, 'azm': 2}
        sensor_data = await client.fetch_last_5_min_data()
        
        if not sensor_data:
            logger.info(f"No recent data synced for user {user_id}")
            return

        # VALIDATION FIX: Ensure sensor_data is a dictionary (type check before .get())
        if not isinstance(sensor_data, dict):
            logger.error(f"FitbitClient returned invalid data type: {type(sensor_data)}. Expected dict. Data: {sensor_data}")
            return

        # Log Fitbit Extraction Success
        logger.info(f"Fitbit: Successfully extracted data for user {user_id}: {sensor_data}")

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

    except Exception as e:
        logger.error(f"Error processing user {user.get('id')}: {str(e)}")

async def scheduled_job():
    """
    Runs every 5 minutes.
    Fetches all active users with devices and processes them.
    """
    logger.info("Starting scheduled data fetch...")
    
    try:
        # Fetch active users who have a device_id
        # FIX: Use .not_.is_('device_id', 'null') instead of .neq('device_id', 'null') for UUID columns
        response = supabase_client.table('users').select('*').not_.is_('device_id', 'null').execute()
        users = response.data
        
        # Log DB User Extraction
        logger.info(f"DB: Extracted {len(users) if users else 0} active users.")

        if not users:
            logger.info("No active users with devices found.")
            return

        # In production, you might want to use a task queue (Celery) here for parallelism
        for user in users:
            await process_user_device(user)
            
    except Exception as e:
        logger.error(f"DB Error during user fetch: {e}")

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
    features = [request.hr, request.hrv, request.azm]
    prediction = predictor.predict(features)
    return {
        "input": {
            "hr": request.hr,
            "hrv": request.hrv,
            "azm": request.azm
        },
        "prediction": prediction
    }