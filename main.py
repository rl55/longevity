import asyncio
import pandas as pd
import json
import math
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD DATA (Make sure filenames match exactly)
# We merge them on timestamp for a synchronized stream
def load_data():
    try:
        hr = pd.read_csv("data/heart_rate.csv")
        hrv = pd.read_csv("data/hrv.csv")
        gluc = pd.read_csv("data/glucose.csv")
        
        # Rename columns for easier merging
        hr = hr.rename(columns={"heart_rate_bpm": "hr", "context": "context"})
        hrv = hrv.rename(columns={"hrv_rmssd_ms": "hrv"})
        gluc = gluc.rename(columns={"glucose_mg_dl": "glucose"})
        
        # Merge (inner join to keep synchronized timepoints)
        # Using simple merge for demo purposes - assuming timestamps align reasonably well
        df = pd.merge(hr[['timestamp', 'hr', 'context']], hrv[['timestamp', 'hrv']], on='timestamp', how='inner')
        df = pd.merge(df, gluc[['timestamp', 'glucose']], on='timestamp', how='left')
        
        # Fill NaN glucose (since it has lower sampling rate) with previous value
        df['glucose'] = df['glucose'].ffill()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

DATASET = load_data()

@app.websocket("/ws/stream/{persona_id}")
async def websocket_endpoint(websocket: WebSocket, persona_id: str):
    await websocket.accept()
    
    # SIMULATION PARAMETERS
    # We simulate the "Resilience Score" (Lambda) based on HRV context
    # High Stress = Low Resilience (0.2), Sleep = High Resilience (0.8)
    current_lambda = 0.5 
    
    try:
        for index, row in DATASET.iterrows():
            # Artificial delay for playback (adjust speed here)
            await asyncio.sleep(0.1) 
            
            # Simple logic to generate the "Hidden Physics" metric for the demo
            context = str(row['context'])
            if "stress" in context:
                target_lambda = 0.2
            elif "sleep" in context:
                target_lambda = 0.9
            else:
                target_lambda = 0.6
            
            # Smooth the transition of lambda so the graph looks scientific, not jumpy
            current_lambda += (target_lambda - current_lambda) * 0.1
            
            # Handle noise (The "Intelligence" Layer)
            is_noise = False
            hr_val = float(row['hr'])
            if hr_val < 30 or hr_val > 200:
                is_noise = True
                # Send a 'cleaned' version for the AI view
                hr_cleaned = 80.0 
            else:
                hr_cleaned = hr_val

            payload = {
                "timestamp": row['timestamp'],
                "raw": {
                    "hr": hr_val,
                    "is_noise": is_noise
                },
                "processed": {
                    "hr_cleaned": hr_cleaned,
                    "hrv": float(row['hrv']),
                    "glucose": float(row['glucose']) if not math.isnan(row['glucose']) else 100.0,
                    # This is your "Proprietary Metric"
                    "resilience_lambda": round(current_lambda, 2),
                    "context": context
                }
            }
            
            await websocket.send_json(payload)
            
    except Exception as e:
        print(f"Stream Error: {e}")