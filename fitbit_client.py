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
        logger.info(f"🔹 Device {self.device_id} for User {self.user_id}")
        logger.info(f"   Access Token: {self.access_token}")
        logger.info(f"   Expires At:   {self.expires_at}")
        
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

    def _process_series(self, data_points, value_key, is_nested=False):
        """
        Calculates metric based on user rule:
        - If outlier exists (extreme high/low), use that value.
        - Else, use average.
        """
        if not data_points:
            return 0
        
        values = []
        for point in data_points:
            if is_nested:
                # Handling HRV/AZM structure: value -> key
                val_obj = point.get('value', {})
                # Ensure val_obj is a dict before .get()
                if isinstance(val_obj, dict):
                    val = val_obj.get(value_key, 0)
                else:
                    val = val_obj # Fallback if it's a direct value
            else:
                val = point.get('value', 0)
            values.append(val)
            
        if not values:
            return 0
            
        np_values = np.array(values)
        
        # Outlier Detection Logic
        if value_key == 'heart_rate':
            max_val = np.max(np_values)
            min_val = np.min(np_values)
            if max_val > self.HR_HIGH_THRESHOLD:
                return float(max_val)
            if min_val < self.HR_LOW_THRESHOLD:
                return float(min_val)
                
        if value_key == 'rmssd': # HRV
            min_val = np.min(np_values)
            if min_val < self.HRV_LOW_THRESHOLD:
                return float(min_val)

        return float(np.mean(np_values))

    async def fetch_last_5_min_data(self):
        # 1. Ensure Token is Valid (Initial check based on time)
        await self._refresh_token_if_needed()
        
        # 2. Setup Time Window
        now = datetime.now()
        end_time = now
        start_time = now - timedelta(minutes=5)
        
        date_str = now.strftime("%Y-%m-%d")
        start_time_str = start_time.strftime("%H:%M")
        end_time_str = end_time.strftime("%H:%M")
        
        async with httpx.AsyncClient() as client:
            # Heart Rate
            hr_url = f"https://api.fitbit.com/1/user/-/activities/heart/date/{date_str}/1d/1min/time/{start_time_str}/{end_time_str}.json"
            hr_resp = await self._get_with_retry(client, hr_url)
            
            # HRV (Intraday requires specific handling, often whole day fetch if filtering is hard)
            hrv_url = f"https://api.fitbit.com/1/user/-/hrv/date/{date_str}/all.json"
            hrv_resp = await self._get_with_retry(client, hrv_url)
            
            # Active Zone Minutes
            azm_url = f"https://api.fitbit.com/1/user/-/activities/active-zone-minutes/date/{date_str}/1d/1min/time/{start_time_str}/{end_time_str}.json"
            azm_resp = await self._get_with_retry(client, azm_url)

        # 4. Process Responses
        if hr_resp.status_code != 200:
            logger.error(f"Fitbit API Error (HR): {hr_resp.status_code} - {hr_resp.text}")
            return None
            
        # Parse HR (Standard dataset structure)
        hr_json = self._safe_get_json(hr_resp)
        hr_data = hr_json.get('activities-heart-intraday', {}).get('dataset', [])
        final_hr = self._process_series(hr_data, 'value', is_nested=False)
        
        # Parse HRV (Nested 'minutes' list structure)
        final_hrv = 0
        if hrv_resp.status_code == 200:
            hrv_json_body = self._safe_get_json(hrv_resp)
            hrv_list = hrv_json_body.get('hrv', [])
            hrv_minutes = []
            
            if isinstance(hrv_list, list):
                 for entry in hrv_list:
                     if isinstance(entry, dict) and 'minutes' in entry:
                         hrv_minutes.extend(entry['minutes'])
            
            relevant_hrv = []
            for item in hrv_minutes:
                if not isinstance(item, dict): continue
                try:
                    t_str = item.get('minute')
                    if not t_str: continue
                    t_obj = datetime.fromisoformat(t_str)
                    if start_time <= t_obj <= end_time:
                        relevant_hrv.append(item)
                except ValueError:
                    continue
            
            final_hrv = self._process_series(relevant_hrv, 'rmssd', is_nested=True)

        # Parse AZM (Nested 'minutes' list structure, similar to HRV)
        final_azm = 0
        if azm_resp.status_code == 200:
            azm_json_body = self._safe_get_json(azm_resp)
            # Correct key based on user input
            azm_list = azm_json_body.get('activities-active-zone-minutes-intraday', [])
            azm_minutes = []

            if isinstance(azm_list, list):
                for entry in azm_list:
                    if isinstance(entry, dict) and 'minutes' in entry:
                        azm_minutes.extend(entry['minutes'])
            
            relevant_azm = []
            for item in azm_minutes:
                if not isinstance(item, dict): continue
                try:
                    t_str = item.get('minute')
                    if not t_str: continue
                    t_obj = datetime.fromisoformat(t_str)
                    
                    if start_time <= t_obj <= end_time:
                        relevant_azm.append(item)
                except ValueError:
                    continue
            
            # Value key is 'activeZoneMinutes' inside the nested 'value' object
            final_azm = self._process_series(relevant_azm, 'activeZoneMinutes', is_nested=True)

        return {
            "hr": final_hr,
            "hrv": final_hrv,
            "azm": final_azm,
            "timestamp": now.isoformat()
        }