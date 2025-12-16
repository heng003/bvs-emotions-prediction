import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

MODEL_PATH = "emotion_model.pkl"

class EmotionPredictor:
    def __init__(self):
        self.model = None
        self.labels = ['Happy', 'Sad', 'Angry', 'Fear']
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("Loaded trained emotion model.")
        else:
            print("No trained model found. Using Heuristic Fallback mode.")
            self.model = None

    def predict(self, features):
        """
        Features: [HeartRate, HRV, ActiveZoneMinutes]
        """
        hr, hrv, azm = features
        
        # 1. Use Trained Model if available
        if self.model:
            # FIX: Create a DataFrame with column names to avoid sklearn UserWarning
            # The model was fitted with feature names, so we must provide them during prediction
            input_df = pd.DataFrame([features], columns=['HR', 'HRV', 'AZM'])
            
            prediction_idx = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            
            # Ensure prediction is within bounds of labels
            if 0 <= prediction_idx < len(self.labels):
                predicted_emotion = self.labels[prediction_idx]
            else:
                predicted_emotion = "Unknown"

            confidence = probabilities[prediction_idx]
            
            return {"emotion": predicted_emotion, "confidence": float(confidence)}

        # 2. Heuristic Fallback (Rule-based if no model trained)
        emotion = "Happy"
        confidence = 0.65 
        
        if hr > 100: 
            if hrv < 20: 
                emotion = "Fear"
                confidence = 0.85
            elif hrv < 40:
                emotion = "Angry"
                confidence = 0.70
            else:
                emotion = "Happy"
        else:
            if hrv < 30:
                emotion = "Sad"
                confidence = 0.80

        return {"emotion": emotion, "confidence": confidence}

# ==========================================
# WESAD PROCESSING & TRAINING
# ==========================================

def process_bvp_signal(bvp_data, fs=64):
    """
    Extracts HR and HRV (RMSSD) from raw BVP signal (PPG).
    fs: Sampling frequency of Empatica E4 BVP is 64Hz.
    """
    # 1. Find Peaks (Heart beats)
    # distance=fs/2 means max 120bpm, roughly. Adjust for higher HR.
    peaks, _ = find_peaks(bvp_data, distance=fs*0.4, prominence=5)
    
    if len(peaks) < 2:
        return 0, 0

    # 2. Calculate RR intervals (in milliseconds)
    rr_intervals = np.diff(peaks) / fs * 1000 
    
    # 3. Calculate HR
    if len(rr_intervals) > 0:
        avg_hr = 60 / (np.mean(rr_intervals) / 1000)
        
        # 4. Calculate RMSSD (Root Mean Square of Successive Differences) - Standard HRV metric
        diff_rr = np.diff(rr_intervals)
        if len(diff_rr) > 0:
            rmssd = np.sqrt(np.mean(diff_rr ** 2))
        else:
            rmssd = 0
    else:
        avg_hr = 0
        rmssd = 0
    
    return avg_hr, rmssd

def load_wesad_data(root_dir="WESAD"):
    print("Loading WESAD Dataset...")
    features = []
    labels = []
    
    # Iterate over Subject folders (S2, S3, etc.)
    if not os.path.exists(root_dir):
        print(f"Warning: {root_dir} folder not found.")
        return pd.DataFrame(), pd.Series()

    subjects = [d for d in os.listdir(root_dir) if d.startswith('S') and os.path.isdir(os.path.join(root_dir, d))]
    
    for subj in subjects:
        # Determine Path: Check for "E4" OR "SX_E4_Data"
        e4_path = os.path.join(root_dir, subj, "E4")
        if not os.path.exists(e4_path):
             e4_path = os.path.join(root_dir, subj, f"{subj}_E4_Data")
        
        if not os.path.exists(e4_path):
            print(f"  Skipping {subj}: Data folder not found (Checked 'E4' and '{subj}_E4_Data').")
            continue
            
        print(f"Processing {subj} found at {e4_path}...")
        
        try:
            # Load CSVs
            # BVP.csv and ACC.csv usually exist in the E4 folder
            bvp_path = os.path.join(e4_path, "BVP.csv")
            acc_path = os.path.join(e4_path, "ACC.csv")

            if not os.path.exists(bvp_path) or not os.path.exists(acc_path):
                print(f"  Missing BVP.csv or ACC.csv in {e4_path}")
                continue

            bvp_df = pd.read_csv(bvp_path, skiprows=2, header=None, names=['val'])
            acc_df = pd.read_csv(acc_path, skiprows=2, header=None, names=['x', 'y', 'z'])
            
            # Load Labels
            lbl_path = os.path.join(e4_path, "labels.csv")
            if os.path.exists(lbl_path):
                # Assuming labels.csv has no header, just label column
                lbl_df = pd.read_csv(lbl_path, header=None, names=['label'])
            else:
                print(f"  No labels.csv found for {subj}. Ensure you have extracted labels from S{subj}.pkl to csv.")
                continue

            # WINDOWING
            # We split data into 1-minute windows to mimic Fitbit Intraday
            window_sec = 60
            bvp_fs = 64
            acc_fs = 32
            
            # Use the shortest length
            n_windows = min(len(bvp_df) // (bvp_fs * window_sec), len(acc_df) // (acc_fs * window_sec))
            
            for i in range(n_windows):
                # 1. Extract Window Data
                bvp_win = bvp_df['val'].iloc[i*bvp_fs*window_sec : (i+1)*bvp_fs*window_sec].values
                acc_win = acc_df.iloc[i*acc_fs*window_sec : (i+1)*acc_fs*window_sec]
                
                # 2. Process BVP -> HR & HRV
                hr, hrv = process_bvp_signal(bvp_win, fs=bvp_fs)
                
                # 3. Process ACC -> Active Zone (Proxy: Mean Magnitude)
                acc_mag = np.sqrt(acc_win['x']**2 + acc_win['y']**2 + acc_win['z']**2)
                azm_proxy = np.mean(acc_mag) / 64.0 # Normalize roughly (1g = 64 units in E4 raw)
                azm_val = 1 if azm_proxy > 1.2 else 0 # Simple threshold: >1.2g avg = Active
                
                # 4. Get Label (Mode of the window)
                idx_start = i * bvp_fs * window_sec
                # Assumption: labels.csv aligns with BVP or is high frequency (700hz) or 1hz.
                lbl_freq_guess = 1
                if len(lbl_df) > len(bvp_df) * 10: # Likely 700Hz
                    lbl_freq_guess = 700 / 64
                elif len(lbl_df) < len(bvp_df) / 10: # Likely 1Hz or less
                    lbl_freq_guess = 1 / 64
                
                lbl_idx = int(idx_start * lbl_freq_guess)
                
                if lbl_idx < len(lbl_df):
                     mode_label = lbl_df['label'].iloc[lbl_idx]
                else:
                    break 

                # WESAD Label Mapping
                # 0=Undefined, 1=Baseline, 2=Stress, 3=Amusement, 4=Meditation
                target_emotion = None
                
                if mode_label == 1: # Baseline -> Sad (Low Arousal)
                    target_emotion = 1 
                elif mode_label == 2: # Stress -> Fear (High Arousal, Negative)
                    target_emotion = 3 
                elif mode_label == 3: # Amusement -> Happy
                    target_emotion = 0 
                elif mode_label == 4: # Meditation -> Happy (Calm)
                    target_emotion = 0 
                
                if target_emotion is not None:
                    features.append([hr, hrv, azm_val])
                    labels.append(target_emotion)

        except Exception as e:
            print(f"Error processing {subj}: {e}")
            continue

    return pd.DataFrame(features, columns=['HR', 'HRV', 'AZM']), pd.Series(labels)

def train_model():
    print("Starting Model Training...")
    
    X, y = load_wesad_data()
    
    if X.empty:
        print("No valid WESAD data found/loaded. Using SYNTHETIC data for model generation.")
        
        # Generate SYNTHETIC samples so user receives a working model.pkl
        # Mapping: 0:Happy, 1:Sad, 2:Angry, 3:Fear
        data = {
            'HR': [], 'HRV': [], 'AZM': [], 'Label': []
        }
        
        for _ in range(2000):
            # Happy (High HRV, Moderate HR)
            data['HR'].append(np.random.normal(75, 10))
            data['HRV'].append(np.random.normal(60, 15))
            data['AZM'].append(np.random.randint(0, 5))
            data['Label'].append(0) # Happy
            
            # Sad (Low HR, Low HRV)
            data['HR'].append(np.random.normal(60, 5))
            data['HRV'].append(np.random.normal(25, 10))
            data['AZM'].append(0)
            data['Label'].append(1) # Sad
            
            # Angry (High HR, Low HRV)
            data['HR'].append(np.random.normal(110, 15))
            data['HRV'].append(np.random.normal(20, 5))
            data['AZM'].append(np.random.randint(5, 20))
            data['Label'].append(2) # Angry
            
            # Fear (Very High HR, Very Low HRV)
            data['HR'].append(np.random.normal(130, 20))
            data['HRV'].append(np.random.normal(15, 5))
            data['AZM'].append(np.random.randint(10, 30))
            data['Label'].append(3) # Fear
            
        df = pd.DataFrame(data)
        X = df[['HR', 'HRV', 'AZM']]
        y = df['Label']

    print(f"Training on {len(X)} samples...")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    acc = clf.score(X_test, y_test)
    print(f"Model Accuracy: {acc:.2f}")
    
    # Save
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()