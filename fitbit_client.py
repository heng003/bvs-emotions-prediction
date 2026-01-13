import httpx
import logging
from datetime import datetime, timedelta, timezone
import numpy as np
from database import supabase_client
import base64
import os

logger = logging.getLogger(__name__)

class FitbitClient:
    def __init__(self, device_record, user_id):
        self.device_id = device_record['id']
        self.user_id = user_id
        self.access_token = device_record['fitbit_access_token']
        self.refresh_token = device_record['fitbit_refresh_token']
        
        # Handle expiry string format (some DBs return 'Z', some '+00:00')
        expires_at_str = device_record['fitbit_expires_at']
        if expires_at_str and expires_at_str.endswith('Z'):
            expires_at_str = expires_at_str[:-1] + '+00:00'
        
        # Fallback if expires_at is somehow missing
        if expires_at_str:
            self.expires_at = datetime.fromisoformat(expires_at_str)
        else:
            self.expires_at = datetime.now(timezone.utc)

        # LOGGING: Print Token and Expiry for debugging
        logger.info(f"Device {self.device_id} for User {self.user_id}")
        logger.info(f"Refresh Token Expires At: {self.expires_at}")
        
        # Load Credentials from Environment
        self.client_id = os.getenv("FITBIT_CLIENT_ID")
        self.client_secret = os.getenv("FITBIT_CLIENT_SECRET")
        
        # Define Outlier Thresholds
        self.HR_HIGH_THRESHOLD = 140
        self.HR_LOW_THRESHOLD = 40
        self.HRV_LOW_THRESHOLD = 10 

    async def _refresh_token_if_needed(self, force=False):
        """Checks expiration and refreshes token if needed."""
        # Refresh if expired or expiring in next 5 mins OR if force is True
        # Use simple UTC comparison
        tz = self.expires_at.tzinfo
        now_utc = datetime.now(tz)
        
        # Use 1 minute for local testing
        # if force or now_utc >= (self.expires_at - timedelta(minutes=1)):
        if force or now_utc >= (self.expires_at - timedelta(minutes=5)):
            logger.info(f"Refreshing token for device {self.device_id} (Force: {force})")
            
            # 1. Construct Basic Auth Header exactly as requested
            auth_str = f"{self.client_id}:{self.client_secret}"
            basic_auth = base64.b64encode(auth_str.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {basic_auth}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.fitbit.com/oauth2/token",
                    headers=headers,
                    data=data
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    # Defensive check for token_data type
                    if not isinstance(token_data, dict):
                        logger.error(f"Token refresh response is not a dict: {token_data}")
                        raise Exception("Invalid Token Response Format")

                    self.access_token = token_data['access_token']
                    self.refresh_token = token_data['refresh_token']
                    
                    # Calculate new expiry (Fitbit returns seconds in 'expires_in')
                    expires_in = token_data.get('expires_in', 3600)
                    new_expiry = datetime.now(tz) + timedelta(seconds=expires_in)
                    
                    # Update DB
                    supabase_client.table('devices').update({
                        "fitbit_access_token": self.access_token,
                        "fitbit_refresh_token": self.refresh_token,
                        "fitbit_expires_at": new_expiry.isoformat()
                    }).eq('id', self.device_id).execute()
                    
                    logger.info("Token refreshed successfully.")
                    # Update local expiry for subsequent calls in same instance
                    self.expires_at = new_expiry
                else:
                    logger.error(f"Failed to refresh token: {response.text}")
                    # In production, you might want to mark device as 'inactive' here
                    raise Exception("Token Refresh Failed")

    async def _get_with_retry(self, client, url):
        """
        Helper function to perform GET requests with automatic 401 retry logic.
        If a 401 (Unauthorized) is received, it forces a token refresh and retries once.
        """
        # 1. Prepare Header
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # 2. First Attempt
        response = await client.get(url, headers=headers)
        
        # DEBUG LOGGING (Safe version)
        logger.info(f"GET {url} returned status {response.status_code}")
        
        # FIX: Removed logger.info(f"Response data: {response.data}") to prevent crash
        if response.status_code != 200:
            logger.warning(f"Response text: {response.text}")
        
        # 3. Handle 401 (Token potentially expired/revoked despite local timestamp)
        if response.status_code == 401:
            logger.warning(f"401 Unauthorized for {url}. Attempting forced token refresh...")
            try:
                # Force Refresh
                await self._refresh_token_if_needed(force=True)
                
                # Retry with new token
                headers = {"Authorization": f"Bearer {self.access_token}"}
                response = await client.get(url, headers=headers)
                logger.info(f"Retry GET {url} returned status {response.status_code}")
                
            except Exception as e:
                logger.error(f"Retry failed due to refresh error: {e}")
                # We return the original 401 response or let the exception bubble up
                # depending on preference. Here we let the caller see the failed response.
        
        return response

    def _safe_get_json(self, response):
        """
        Helper to safely parse JSON. 
        If response is list or parsing fails, returns empty dict to prevent AttributeError.
        """
        try:
            data = response.json()
            if isinstance(data, list):
                logger.warning(f"Fitbit API returned a LIST instead of DICT for url: {response.url}. Data: {data}")
                return {} # Return empty dict to safely call .get() on
            return data
        except Exception as e:
            logger.error(f"Failed to parse JSON for {response.url}: {e}")
            return {}

    def _filter_data_with_fallback(self, parsed_data, start_time, end_time, metric_name="Data", fallback_limit_minutes=5):
        """
        Generic fallback logic:
        1. Try to find data strictly within [start_time, end_time].
        2. If empty, find the closest data point <= end_time, BUT within limit.
           Max acceptable age = window size (5 min) + fallback_limit
        """
        relevant_items = []
        
        # 1. Strict Window
        for t_obj, item in parsed_data:
            if start_time <= t_obj <= end_time:
                relevant_items.append(item)
                
        # 2. Fallback
        if not relevant_items and parsed_data:
            closest_timestamp = None
            closest_item = None
            
            # Allow data up to (window_duration + fallback_limit) ago from end_time
            # window duration is implicit (start_time - end_time), but we know it's 5 mins.
            # So start_time is (end_time - 5min). 
            # We want to go back `fallback_limit` BEFORE start_time.
            cutoff_time = start_time - timedelta(minutes=fallback_limit_minutes)

            for t_obj, item in parsed_data:
                # Must be before end_time AND after cutoff
                if cutoff_time <= t_obj <= end_time:
                    if closest_timestamp is None or t_obj > closest_timestamp:
                        closest_timestamp = t_obj
                        closest_item = item
            
            if closest_item:
                logger.info(f"{metric_name} missing for window. Fallback to closest data at {closest_timestamp} (Limit: {fallback_limit_minutes}m)")
                relevant_items.append(closest_item)
            else:
                logger.warning(f"{metric_name} missing in window AND within fallback limit ({fallback_limit_minutes}m).")

        return relevant_items

    def _process_series(self, data_list, value_key, is_nested=False):
        """
        Calculates the average of a specific metric from a list of data items.
        Handles nested 'value' dictionaries if is_nested=True.
        """
        if not data_list:
            return 0
            
        values = []
        for item in data_list:
            try:
                if is_nested:
                    # e.g. item['value']['rmssd']
                    val_dict = item.get('value', {})
                    if isinstance(val_dict, dict):
                        val = val_dict.get(value_key)
                        if val is not None:
                            values.append(float(val))
                else:
                    # e.g. item['value']
                    val = item.get(value_key)
                    if val is not None:
                        values.append(float(val))
            except (ValueError, TypeError):
                continue
                
        if not values:
            return 0
            
        return float(np.mean(values))

    async def fetch_last_5_min_data(self):
        # Define Asia/Kuala_Lumpur Timezone (UTC+8)
        KL_TZ = timezone(timedelta(hours=8))

        # 1. Ensure Token is Valid
        await self._refresh_token_if_needed()
        
        # 2. Setup Time Window
        # Use KL Time for querying Fitbit (API expects user local time)
        # Robust conversion: UTC -> KL
        now = datetime.now(timezone.utc).astimezone(KL_TZ)
        end_time = now
        start_time = now - timedelta(minutes=5)
        
        # Fetch from midnight to ensure we have previous data for fallback
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        date_str = now.strftime("%Y-%m-%d")
        
        # API requires HH:MM for time params
        start_request_str = midnight.strftime("%H:%M") 
        end_request_str = end_time.strftime("%H:%M")
        
        async with httpx.AsyncClient() as client:
            # Heart Rate - Fetch from midnight
            hr_url = f"https://api.fitbit.com/1/user/-/activities/heart/date/{date_str}/1d/1min/time/{start_request_str}/{end_request_str}.json"
            hr_resp = await self._get_with_retry(client, hr_url)
            
            # HRV - Fetches all day by default usually
            hrv_url = f"https://api.fitbit.com/1/user/-/hrv/date/{date_str}/all.json"
            hrv_resp = await self._get_with_retry(client, hrv_url)
            
            # Active Zone Minutes - Fetch from midnight
            azm_url = f"https://api.fitbit.com/1/user/-/activities/active-zone-minutes/date/{date_str}/1d/1min/time/{start_request_str}/{end_request_str}.json"
            azm_resp = await self._get_with_retry(client, azm_url)

        # 4. Process Responses
        # --- Heart Rate ---
        final_hr = 0
        if hr_resp.status_code == 200:
            hr_json = self._safe_get_json(hr_resp)
            hr_dataset = hr_json.get('activities-heart-intraday', {}).get('dataset', [])
            
            # Parse HR Timestamps (Time is "HH:MM:SS", need to combine with date)
            parsed_hr = []
            for item in hr_dataset:
                t_str = item.get('time')
                if not t_str: continue
                try:
                    # Combine date_str YYYY-MM-DD + time HH:MM:SS
                    full_dt_str = f"{date_str} {t_str}"
                    t_obj = datetime.strptime(full_dt_str, "%Y-%m-%d %H:%M:%S")
                    # Make aware (KL Time)
                    t_obj = t_obj.replace(tzinfo=KL_TZ)
                    parsed_hr.append((t_obj, item))
                except ValueError:
                    continue
            
            # HR Fallback Limit: 5 min
            relevant_hr = self._filter_data_with_fallback(parsed_hr, start_time, end_time, "HeartRate", fallback_limit_minutes=5)
            final_hr = self._process_series(relevant_hr, 'value', is_nested=False)
        else:
            logger.error(f"Fitbit API Error (HR): {hr_resp.status_code} - {hr_resp.text}")

        # --- HRV ---
        final_hrv = 0
        if hrv_resp.status_code == 200:
            hrv_json_body = self._safe_get_json(hrv_resp)
            hrv_list = hrv_json_body.get('hrv', [])
            hrv_minutes = []
            if isinstance(hrv_list, list):
                 for entry in hrv_list:
                     if isinstance(entry, dict) and 'minutes' in entry:
                         hrv_minutes.extend(entry['minutes'])
            
            parsed_hrv = []
            for item in hrv_minutes:
                if not isinstance(item, dict): continue
                try:
                    t_str = item.get('minute')
                    if not t_str: continue
                    t_obj = datetime.fromisoformat(t_str)
                    # Make aware (KL Time) if naive
                    if t_obj.tzinfo is None:
                        t_obj = t_obj.replace(tzinfo=KL_TZ)
                    parsed_hrv.append((t_obj, item))
                except ValueError:
                    continue
            
            # HRV Fallback Limit: 60 min
            relevant_hrv = self._filter_data_with_fallback(parsed_hrv, start_time, end_time, "HRV", fallback_limit_minutes=60)
            final_hrv = self._process_series(relevant_hrv, 'rmssd', is_nested=True)

        # --- AZM ---
        final_azm = 0
        if azm_resp.status_code == 200:
            azm_json_body = self._safe_get_json(azm_resp)
            azm_list = azm_json_body.get('activities-active-zone-minutes-intraday', [])
            azm_minutes = []

            if isinstance(azm_list, list):
                for entry in azm_list:
                    if isinstance(entry, dict) and 'minutes' in entry:
                        azm_minutes.extend(entry['minutes'])
            
            parsed_azm = []
            for item in azm_minutes:
                if not isinstance(item, dict): continue
                try:
                    t_str = item.get('minute')
                    if not t_str: continue
                    t_obj = datetime.fromisoformat(t_str)
                    # Make aware (KL Time) if naive
                    if t_obj.tzinfo is None:
                        t_obj = t_obj.replace(tzinfo=KL_TZ)
                    parsed_azm.append((t_obj, item))
                except ValueError:
                    continue
            
            # AZM Fallback Limit: 5 min
            relevant_azm = self._filter_data_with_fallback(parsed_azm, start_time, end_time, "AZM", fallback_limit_minutes=5)
            final_azm = self._process_series(relevant_azm, 'activeZoneMinutes', is_nested=True)

        # Check if ALL data is missing
        if final_hr == 0 and final_hrv == 0 and final_azm == 0:
            logger.info("No valid data found for HR, HRV, or AZM (even with fallback). skipping prediction.")
            return None

        return {
            "hr": final_hr,
            "hrv": final_hrv,
            "azm": final_azm,
            "timestamp": now.isoformat()
        }