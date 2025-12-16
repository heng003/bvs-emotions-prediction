Body Vital Signs Emotion Prediction System

This project is a FastAPI-based backend system that periodically fetches intraday health data (Heart Rate, HRV, Active Zone Minutes) from the Fitbit API, processes it to detect outliers, and uses a Machine Learning model (Random Forest) to predict the user's emotional state (Happy, Sad, Angry, Fear). Results are stored in a Supabase database.

Project Structure

fitbit-emotion-system/
│
├── main.py              # Entry point: FastAPI app, Scheduler, and core logic orchestration
├── database.py          # Database connection setup (Supabase)
├── fitbit_client.py     # Service class for Fitbit API interactions & Token refreshing
├── emotion_model.py     # Machine Learning model definition, inference logic, and training script
├── requirements.txt     # List of Python dependencies
└── .env                 # Environment variables (Credentials)


Prerequisites

Python 3.9+

Supabase Account: You need a project with the users, devices, and bvs_emotion tables created as per the SQL schema provided.

Fitbit Developer Account: You need a registered application to get a Client ID and Client Secret.

Installation

Clone the repository (or create the directory structure above).

Install Dependencies:
Create a requirements.txt file with the content below, then install:

pip install fastapi uvicorn supabase apscheduler httpx pandas scikit-learn numpy python-dotenv joblib


Environment Setup:
Create a file named .env in the root directory and add your credentials:

# Supabase Credentials
SUPABASE_URL=[https://your-project-id.supabase.co](https://your-project-id.supabase.co)
SUPABASE_KEY=your-service-role-key-or-anon-key

# Fitbit Application Credentials
FITBIT_CLIENT_ID=your_fitbit_client_id
FITBIT_CLIENT_SECRET=your_fitbit_client_secret


Usage

1. Training the Model (Optional but Recommended)

By default, the system uses a heuristic (rule-based) fallback if no trained model is found. To train a Random Forest classifier:

Open emotion_model.py.

(Optional) Replace the synthetic data generation in train_model() with real loaded data (e.g., from WESAD dataset).

Run the script directly:

python emotion_model.py


This will create a file named emotion_model.pkl in your directory.

2. Running the Server

Start the FastAPI server. This will automatically start the background scheduler which runs every 5 minutes.

uvicorn main:app --reload


Server URL: http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs (for testing endpoints)

3. How it Works

Scheduler: Every 5 minutes, main.py triggers scheduled_job().

User Lookup: It queries Supabase for users who have a linked device_id.

Data Fetch: fitbit_client.py uses the stored Access Token to fetch the last 5 minutes of data.

If the token is expired, it automatically refreshes it using the Refresh Token and updates Supabase.

Data Processing:

Calculates averages for Heart Rate and HRV.

Outlier Logic: If a specific minute has an extreme value (e.g., HR > 140 or HRV < 10), that extreme value is used instead of the average to capture sudden stress events.

Prediction: The emotion_model.py takes the processed metrics and predicts the emotion (Happy, Sad, Angry, Fear) and a confidence score.

Storage: The result is inserted into the bvs_emotion table in Supabase.

API Endpoints

GET /: Checks system status.

POST /trigger-now: Manually triggers the data fetch/prediction cycle (useful for testing without waiting 5 minutes).

Troubleshooting

Fitbit 429 Errors (Rate Limits): The system fetches 3 endpoints (HR, HRV, AZM) per user every 5 minutes. This is generally within Fitbit's standard rate limits (150 calls/hour per user), but be mindful if testing aggressively.

Token Errors: Ensure the initial fitbit_access_token and fitbit_refresh_token in your Supabase devices table are valid before starting the server.