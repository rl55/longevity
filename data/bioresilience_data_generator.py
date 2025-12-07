"""
BioResilienceOS Synthetic Data Generator
Generates Tier 1 (Wearables) and Tier 2 (CGM) time series data for four personas
with context-adaptive sampling intervals and intentional outliers for pipeline testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os

np.random.seed(42)

@dataclass
class PersonaConfig:
    """Configuration for each persona's physiological parameters."""
    name: str
    lambda_hr: float          # HR recovery rate (restoring force)
    lambda_hrv: float         # HRV stability
    lambda_glucose: float     # Glucose clearance rate
    
    # Baseline values
    resting_hr: float
    hr_variability: float
    hrv_rmssd_baseline: float
    hrv_variability: float
    glucose_baseline: float
    glucose_variability: float
    resp_rate_baseline: float
    skin_temp_baseline: float
    
    # Recovery characteristics
    hr_recovery_noise: float
    glucose_meal_spike: float
    sleep_efficiency: float
    
    # Outlier probability (higher = more sensor issues)
    outlier_prob: float

# Define the four personas from the presentation
PERSONAS = {
    'iron_grandmother': PersonaConfig(
        name='Iron Grandmother (70yo Athlete)',
        lambda_hr=0.8,
        lambda_hrv=0.75,
        lambda_glucose=0.7,
        resting_hr=55,
        hr_variability=8,
        hrv_rmssd_baseline=45,
        hrv_variability=12,
        glucose_baseline=92,
        glucose_variability=8,
        resp_rate_baseline=14,
        skin_temp_baseline=36.2,
        hr_recovery_noise=3,
        glucose_meal_spike=35,
        sleep_efficiency=0.88,
        outlier_prob=0.005
    ),
    'stressed_executive': PersonaConfig(
        name='Stressed Executive (40yo, No Sleep)',
        lambda_hr=0.2,
        lambda_hrv=0.15,
        lambda_glucose=0.25,
        resting_hr=72,
        hr_variability=15,
        hrv_rmssd_baseline=28,
        hrv_variability=18,
        glucose_baseline=105,
        glucose_variability=18,
        resp_rate_baseline=17,
        skin_temp_baseline=36.5,
        hr_recovery_noise=8,
        glucose_meal_spike=55,
        sleep_efficiency=0.65,
        outlier_prob=0.008
    ),
    'menopausal_transition': PersonaConfig(
        name='Menopausal Transition (50yo)',
        lambda_hr=0.45,  # Variable - will add fluctuations
        lambda_hrv=0.35,
        lambda_glucose=0.4,
        resting_hr=68,
        hr_variability=12,
        hrv_rmssd_baseline=32,
        hrv_variability=20,  # Higher variability - system instability
        glucose_baseline=98,
        glucose_variability=14,
        resp_rate_baseline=15,
        skin_temp_baseline=36.8,  # Slightly elevated, hot flashes
        hr_recovery_noise=6,
        glucose_meal_spike=48,
        sleep_efficiency=0.72,
        outlier_prob=0.006
    ),
    'frail_baseline': PersonaConfig(
        name='Frail Baseline (80yo)',
        lambda_hr=0.05,
        lambda_hrv=0.08,
        lambda_glucose=0.12,
        resting_hr=68,
        hr_variability=6,  # Low variability - "flatline"
        hrv_rmssd_baseline=18,
        hrv_variability=5,  # Very low - lost adaptive capacity
        glucose_baseline=115,
        glucose_variability=12,
        resp_rate_baseline=18,
        skin_temp_baseline=35.8,  # Lower baseline
        hr_recovery_noise=4,
        glucose_meal_spike=65,
        sleep_efficiency=0.60,
        outlier_prob=0.012  # More sensor issues
    )
}

def simulate_ou_process(
    n_points: int,
    dt_seconds: float,
    lambda_param: float,
    mu: float,
    sigma: float,
    initial_value: float = None
) -> np.ndarray:
    """
    Simulate Ornstein-Uhlenbeck process.
    dX = -λ(X - μ)dt + σdW
    
    Higher λ = faster return to baseline (more resilient)
    λ is in units of per-hour, so we convert dt to hours.
    """
    if initial_value is None:
        initial_value = mu
    
    # Convert dt from seconds to hours for consistent λ units
    dt_hours = dt_seconds / 3600.0
    
    values = np.zeros(n_points)
    values[0] = initial_value
    
    for i in range(1, n_points):
        # Exact discretization of OU process for numerical stability
        # X(t+dt) = μ + (X(t) - μ) * exp(-λ*dt) + σ*sqrt((1-exp(-2λdt))/(2λ)) * N(0,1)
        decay = np.exp(-lambda_param * dt_hours)
        
        # Handle case where lambda is very small
        if lambda_param > 0.001:
            noise_scale = sigma * np.sqrt((1 - np.exp(-2 * lambda_param * dt_hours)) / (2 * lambda_param))
        else:
            # For small lambda, use approximation
            noise_scale = sigma * np.sqrt(dt_hours)
        
        values[i] = mu + (values[i-1] - mu) * decay + noise_scale * np.random.randn()
    
    return values

def generate_activity_schedule(start_time: datetime, days: int = 7) -> pd.DataFrame:
    """
    Generate a realistic activity schedule with context states.
    States: sleep, resting_awake, active_low, active_high, post_exercise, post_meal, stress_event
    """
    schedule = []
    current_time = start_time
    end_time = start_time + timedelta(days=days)
    
    while current_time < end_time:
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        # Determine activity context based on time of day
        if 23 <= hour or hour < 6:
            context = 'sleep'
            duration_minutes = np.random.randint(30, 90)
        elif 6 <= hour < 7:
            context = 'sleep_transition'
            duration_minutes = np.random.randint(10, 30)
        elif 7 <= hour < 8:
            # Morning routine - might include exercise
            if np.random.random() < 0.3:
                context = 'active_high'
                duration_minutes = np.random.randint(20, 45)
            else:
                context = 'resting_awake'
                duration_minutes = np.random.randint(15, 45)
        elif 8 <= hour < 9:
            context = 'post_meal'  # Breakfast
            duration_minutes = np.random.randint(30, 60)
        elif 9 <= hour < 12:
            # Work hours
            if np.random.random() < 0.15:
                context = 'stress_event'
                duration_minutes = np.random.randint(10, 30)
            elif np.random.random() < 0.3:
                context = 'active_low'
                duration_minutes = np.random.randint(10, 30)
            else:
                context = 'resting_awake'
                duration_minutes = np.random.randint(30, 90)
        elif 12 <= hour < 14:
            context = 'post_meal'  # Lunch
            duration_minutes = np.random.randint(45, 90)
        elif 14 <= hour < 17:
            # Afternoon
            if np.random.random() < 0.1:
                context = 'stress_event'
                duration_minutes = np.random.randint(10, 30)
            elif np.random.random() < 0.2:
                context = 'active_low'
                duration_minutes = np.random.randint(15, 45)
            else:
                context = 'resting_awake'
                duration_minutes = np.random.randint(30, 90)
        elif 17 <= hour < 18:
            # Evening exercise window
            if np.random.random() < 0.4:
                context = 'active_high'
                duration_minutes = np.random.randint(30, 60)
            else:
                context = 'active_low'
                duration_minutes = np.random.randint(20, 40)
        elif 18 <= hour < 19:
            if np.random.random() < 0.5:
                context = 'post_exercise'
                duration_minutes = np.random.randint(15, 30)
            else:
                context = 'resting_awake'
                duration_minutes = np.random.randint(20, 40)
        elif 19 <= hour < 21:
            context = 'post_meal'  # Dinner
            duration_minutes = np.random.randint(60, 120)
        elif 21 <= hour < 23:
            context = 'resting_awake'
            duration_minutes = np.random.randint(30, 60)
        else:
            context = 'resting_awake'
            duration_minutes = 30
        
        schedule.append({
            'start_time': current_time,
            'context': context,
            'duration_minutes': duration_minutes
        })
        
        current_time += timedelta(minutes=duration_minutes)
    
    return pd.DataFrame(schedule)

def get_sampling_interval(context: str, signal_type: str) -> int:
    """
    Return sampling interval in seconds based on context and signal type.
    Implements the adaptive sampling logic from our architecture.
    """
    intervals = {
        'sleep': {
            'heart_rate': 300,      # 5 min
            'hrv': 300,             # 5 min windows
            'respiratory_rate': 300,
            'spo2': 600,            # 10 min
            'skin_temp': 300,
            'motion': 30,
            'glucose': 300          # 5 min (CGM native)
        },
        'sleep_transition': {
            'heart_rate': 30,       # Higher resolution during awakening
            'hrv': 60,
            'respiratory_rate': 60,
            'spo2': 300,
            'skin_temp': 60,
            'motion': 10,
            'glucose': 300
        },
        'resting_awake': {
            'heart_rate': 600,      # 10 min
            'hrv': 900,             # Opportunistic, ~15 min
            'respiratory_rate': 900,
            'spo2': np.nan,         # Not measured during day
            'skin_temp': 3600,      # Hourly
            'motion': 60,
            'glucose': 300
        },
        'active_low': {
            'heart_rate': 120,      # 2 min
            'hrv': np.nan,          # Not reliable during activity
            'respiratory_rate': 120,
            'spo2': np.nan,
            'skin_temp': 1800,
            'motion': 10,
            'glucose': 300
        },
        'active_high': {
            'heart_rate': 5,        # Near-continuous
            'hrv': np.nan,          # Not reliable
            'respiratory_rate': 60,
            'spo2': np.nan,
            'skin_temp': 600,
            'motion': 1,
            'glucose': 300
        },
        'post_exercise': {
            'heart_rate': 5,        # Critical - recovery curve
            'hrv': 60,              # Post-workout window
            'respiratory_rate': 30,
            'spo2': np.nan,
            'skin_temp': 300,
            'motion': 10,
            'glucose': 300
        },
        'post_meal': {
            'heart_rate': 300,
            'hrv': 600,
            'respiratory_rate': 600,
            'spo2': np.nan,
            'skin_temp': 1800,
            'motion': 60,
            'glucose': 300          # CGM captures meal response
        },
        'stress_event': {
            'heart_rate': 30,       # Capture stress response
            'hrv': 60,
            'respiratory_rate': 60,
            'spo2': np.nan,
            'skin_temp': 300,
            'motion': 30,
            'glucose': 300
        }
    }
    
    return intervals.get(context, intervals['resting_awake']).get(signal_type, 300)

def add_outliers(values: np.ndarray, outlier_prob: float, signal_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add realistic outliers based on signal type.
    Returns values and quality flags (1 = good, 0 = outlier/artifact).
    """
    quality = np.ones(len(values))
    outlier_mask = np.random.random(len(values)) < outlier_prob
    
    outlier_types = {
        'heart_rate': [
            (0.3, lambda v: 0),              # Sensor dropout
            (0.2, lambda v: v * 3),          # Motion artifact (spike)
            (0.2, lambda v: np.random.randint(200, 250)),  # Implausible high
            (0.15, lambda v: np.random.randint(20, 35)),   # Implausible low
            (0.15, lambda v: v + np.random.randint(-50, 50))  # Random jump
        ],
        'hrv': [
            (0.4, lambda v: 0),              # Failed measurement
            (0.2, lambda v: v * 4),          # Artifact spike
            (0.2, lambda v: np.random.randint(150, 300)),  # Implausible high
            (0.2, lambda v: max(1, v - 30))  # Sudden drop
        ],
        'respiratory_rate': [
            (0.4, lambda v: 0),
            (0.3, lambda v: np.random.randint(35, 50)),  # Too high
            (0.3, lambda v: np.random.randint(4, 8))     # Too low
        ],
        'spo2': [
            (0.5, lambda v: 0),              # Failed read
            (0.3, lambda v: np.random.randint(70, 85)),  # Sensor slip
            (0.2, lambda v: 100)             # Ceiling
        ],
        'skin_temp': [
            (0.3, lambda v: 0),
            (0.3, lambda v: np.random.uniform(30, 33)),  # Cold contact
            (0.2, lambda v: np.random.uniform(39, 42)),  # Warm contact
            (0.2, lambda v: v + np.random.uniform(-2, 2))
        ],
        'motion': [
            (0.5, lambda v: 0),              # Still period read as dropout
            (0.3, lambda v: np.random.uniform(0, 0.01)), # Near-zero
            (0.2, lambda v: v * 5)           # Motion artifact
        ],
        'glucose': [
            (0.3, lambda v: np.nan),         # Sensor warmup/calibration
            (0.3, lambda v: np.random.randint(40, 55)),  # Compression low
            (0.2, lambda v: np.random.randint(250, 400)), # Implausible high
            (0.2, lambda v: v + np.random.randint(-40, 40))  # Noise spike
        ]
    }
    
    if signal_type in outlier_types:
        outlier_choices = outlier_types[signal_type]
        probs = [c[0] for c in outlier_choices]
        probs = np.array(probs) / sum(probs)
        
        for i in np.where(outlier_mask)[0]:
            choice = np.random.choice(len(outlier_choices), p=probs)
            values[i] = outlier_choices[choice][1](values[i])
            quality[i] = 0
    
    return values, quality

def generate_heart_rate_data(
    schedule: pd.DataFrame,
    persona: PersonaConfig,
    start_time: datetime
) -> pd.DataFrame:
    """Generate heart rate time series with context-adaptive sampling."""
    records = []
    
    for _, row in schedule.iterrows():
        context = row['context']
        interval = get_sampling_interval(context, 'heart_rate')
        duration_sec = row['duration_minutes'] * 60
        n_samples = max(1, int(duration_sec / interval))
        
        # Set parameters based on context
        if context == 'sleep':
            base_hr = persona.resting_hr - 8
            variability = persona.hr_variability * 0.5
            lambda_eff = persona.lambda_hr * 1.2
        elif context in ['active_high']:
            base_hr = persona.resting_hr + 60 + np.random.randint(0, 30)
            variability = persona.hr_variability * 2
            lambda_eff = persona.lambda_hr * 0.5
        elif context == 'active_low':
            base_hr = persona.resting_hr + 25
            variability = persona.hr_variability * 1.3
            lambda_eff = persona.lambda_hr * 0.7
        elif context == 'post_exercise':
            # Recovery curve - starts high, returns to baseline
            base_hr = persona.resting_hr
            variability = persona.hr_variability * 0.8
            lambda_eff = persona.lambda_hr  # This is the key λ we're measuring
        elif context == 'stress_event':
            base_hr = persona.resting_hr + 15
            variability = persona.hr_variability * 1.5
            lambda_eff = persona.lambda_hr * 0.6
        else:
            base_hr = persona.resting_hr
            variability = persona.hr_variability
            lambda_eff = persona.lambda_hr
        
        # For menopausal transition, add λ variability (system instability)
        if persona.name.startswith('Menopausal'):
            lambda_eff *= (0.7 + 0.6 * np.random.random())
        
        # Generate OU process
        if context == 'post_exercise':
            # Special case: recovery from elevated state
            initial_hr = persona.resting_hr + 50 + np.random.randint(0, 20)
            hr_values = simulate_ou_process(
                n_samples, interval, lambda_eff, base_hr,
                persona.hr_recovery_noise, initial_hr
            )
        else:
            hr_values = simulate_ou_process(
                n_samples, interval, lambda_eff, base_hr, variability
            )
        
        # Add outliers
        hr_values, quality = add_outliers(hr_values, persona.outlier_prob, 'heart_rate')
        
        # Create timestamps
        for i in range(n_samples):
            timestamp = row['start_time'] + timedelta(seconds=i * interval)
            records.append({
                'timestamp': timestamp,
                'heart_rate_bpm': max(0, hr_values[i]),
                'context': context,
                'sampling_interval_sec': interval,
                'quality_flag': int(quality[i])
            })
    
    df = pd.DataFrame(records)
    df = df[df['timestamp'] >= start_time]
    return df.sort_values('timestamp').reset_index(drop=True)

def generate_hrv_data(
    schedule: pd.DataFrame,
    persona: PersonaConfig,
    start_time: datetime
) -> pd.DataFrame:
    """Generate HRV (RMSSD) time series with context-adaptive sampling."""
    records = []
    
    for _, row in schedule.iterrows():
        context = row['context']
        interval = get_sampling_interval(context, 'hrv')
        
        if np.isnan(interval):  # Skip contexts where HRV isn't measured
            continue
            
        duration_sec = row['duration_minutes'] * 60
        n_samples = max(1, int(duration_sec / interval))
        
        # Set parameters based on context
        if context == 'sleep':
            base_hrv = persona.hrv_rmssd_baseline * 1.3
            variability = persona.hrv_variability * 0.6
            lambda_eff = persona.lambda_hrv * 1.3
        elif context == 'post_exercise':
            base_hrv = persona.hrv_rmssd_baseline * 0.6
            variability = persona.hrv_variability * 1.2
            lambda_eff = persona.lambda_hrv
        elif context == 'stress_event':
            base_hrv = persona.hrv_rmssd_baseline * 0.7
            variability = persona.hrv_variability * 1.5
            lambda_eff = persona.lambda_hrv * 0.5
        else:
            base_hrv = persona.hrv_rmssd_baseline
            variability = persona.hrv_variability
            lambda_eff = persona.lambda_hrv
        
        # Menopausal variability
        if persona.name.startswith('Menopausal'):
            lambda_eff *= (0.6 + 0.8 * np.random.random())
            variability *= 1.3
        
        hrv_values = simulate_ou_process(
            n_samples, interval, lambda_eff, base_hrv, variability
        )
        hrv_values = np.maximum(hrv_values, 5)  # Floor at 5ms
        
        hrv_values, quality = add_outliers(hrv_values, persona.outlier_prob * 1.5, 'hrv')
        
        for i in range(n_samples):
            timestamp = row['start_time'] + timedelta(seconds=i * interval)
            records.append({
                'timestamp': timestamp,
                'hrv_rmssd_ms': max(0, hrv_values[i]),
                'context': context,
                'sampling_interval_sec': interval,
                'quality_flag': int(quality[i])
            })
    
    df = pd.DataFrame(records)
    df = df[df['timestamp'] >= start_time]
    return df.sort_values('timestamp').reset_index(drop=True)

def generate_glucose_data(
    schedule: pd.DataFrame,
    persona: PersonaConfig,
    start_time: datetime
) -> pd.DataFrame:
    """Generate CGM glucose time series with meal responses."""
    records = []
    
    # CGM samples every 5 minutes regardless of context
    interval = 300  # 5 minutes
    dt_hours = interval / 3600.0
    
    # Generate continuous glucose over the entire period
    total_duration = (schedule['start_time'].max() - schedule['start_time'].min()).total_seconds()
    n_samples = int(total_duration / interval)
    
    glucose = np.zeros(n_samples)
    glucose[0] = persona.glucose_baseline
    
    # Create a meal schedule from post_meal contexts
    meal_times = schedule[schedule['context'] == 'post_meal']['start_time'].tolist()
    
    for i in range(1, n_samples):
        current_time = start_time + timedelta(seconds=i * interval)
        
        # Check if we're in a meal response window (2 hours after meal)
        in_meal_response = False
        meal_progress = 0
        for meal_time in meal_times:
            time_since_meal = (current_time - meal_time).total_seconds() / 60
            if 0 < time_since_meal < 120:  # Within 2 hours
                in_meal_response = True
                meal_progress = time_since_meal / 120
                break
        
        if in_meal_response:
            # Meal response curve: spike then return
            if meal_progress < 0.3:
                # Rising phase
                target = persona.glucose_baseline + persona.glucose_meal_spike * (meal_progress / 0.3)
            else:
                # Recovery phase - this is where λ matters!
                peak = persona.glucose_baseline + persona.glucose_meal_spike
                hours_since_peak = (meal_progress - 0.3) * 2  # Convert to hours
                # Lower λ = slower return
                recovery_factor = 1 - np.exp(-persona.lambda_glucose * hours_since_peak)
                target = peak - (peak - persona.glucose_baseline) * recovery_factor
            
            # OU process toward target using exact discretization
            decay = np.exp(-persona.lambda_glucose * dt_hours)
            noise_scale = persona.glucose_variability * 0.3 * np.sqrt(dt_hours)
            glucose[i] = target + (glucose[i-1] - target) * decay + noise_scale * np.random.randn()
        else:
            # Fasting/baseline - pure OU around baseline
            decay = np.exp(-persona.lambda_glucose * dt_hours)
            if persona.lambda_glucose > 0.001:
                noise_scale = persona.glucose_variability * np.sqrt((1 - np.exp(-2 * persona.lambda_glucose * dt_hours)) / (2 * persona.lambda_glucose))
            else:
                noise_scale = persona.glucose_variability * np.sqrt(dt_hours)
            glucose[i] = persona.glucose_baseline + (glucose[i-1] - persona.glucose_baseline) * decay + noise_scale * np.random.randn()
        
        glucose[i] = max(40, min(400, glucose[i]))
    
    # Add circadian variation
    for i in range(n_samples):
        current_time = start_time + timedelta(seconds=i * interval)
        hour = current_time.hour
        # Dawn phenomenon: glucose rises 4-7am
        if 4 <= hour < 7:
            glucose[i] += 8 * (1 - abs(hour - 5.5) / 1.5)
    
    # Add outliers
    glucose, quality = add_outliers(glucose, persona.outlier_prob, 'glucose')
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(seconds=i * interval)
        # Determine context from schedule
        context = 'unknown'
        for _, row in schedule.iterrows():
            if row['start_time'] <= timestamp < row['start_time'] + timedelta(minutes=row['duration_minutes']):
                context = row['context']
                break
        
        records.append({
            'timestamp': timestamp,
            'glucose_mg_dl': glucose[i] if not np.isnan(glucose[i]) else np.nan,
            'context': context,
            'sampling_interval_sec': interval,
            'quality_flag': int(quality[i]) if not np.isnan(quality[i]) else 0
        })
    
    return pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)

def generate_respiratory_rate_data(
    schedule: pd.DataFrame,
    persona: PersonaConfig,
    start_time: datetime
) -> pd.DataFrame:
    """Generate respiratory rate time series."""
    records = []
    
    for _, row in schedule.iterrows():
        context = row['context']
        interval = get_sampling_interval(context, 'respiratory_rate')
        duration_sec = row['duration_minutes'] * 60
        n_samples = max(1, int(duration_sec / interval))
        
        if context == 'sleep':
            base_rr = persona.resp_rate_baseline - 2
            variability = 1.5
        elif context in ['active_high', 'active_low']:
            base_rr = persona.resp_rate_baseline + 8
            variability = 3
        elif context == 'stress_event':
            base_rr = persona.resp_rate_baseline + 4
            variability = 2.5
        else:
            base_rr = persona.resp_rate_baseline
            variability = 2
        
        rr_values = simulate_ou_process(n_samples, interval, 0.5, base_rr, variability)
        rr_values = np.clip(rr_values, 8, 40)
        
        rr_values, quality = add_outliers(rr_values, persona.outlier_prob, 'respiratory_rate')
        
        for i in range(n_samples):
            timestamp = row['start_time'] + timedelta(seconds=i * interval)
            records.append({
                'timestamp': timestamp,
                'respiratory_rate_bpm': max(0, rr_values[i]),
                'context': context,
                'sampling_interval_sec': interval,
                'quality_flag': int(quality[i])
            })
    
    df = pd.DataFrame(records)
    df = df[df['timestamp'] >= start_time]
    return df.sort_values('timestamp').reset_index(drop=True)

def generate_spo2_data(
    schedule: pd.DataFrame,
    persona: PersonaConfig,
    start_time: datetime
) -> pd.DataFrame:
    """Generate SpO2 data (sleep periods only)."""
    records = []
    
    sleep_periods = schedule[schedule['context'].isin(['sleep', 'sleep_transition'])]
    
    for _, row in sleep_periods.iterrows():
        context = row['context']
        interval = get_sampling_interval(context, 'spo2')
        
        if np.isnan(interval):
            continue
            
        duration_sec = row['duration_minutes'] * 60
        n_samples = max(1, int(duration_sec / interval))
        
        # SpO2 baseline varies by persona
        if persona.name.startswith('Frail'):
            base_spo2 = 94
            variability = 2
        else:
            base_spo2 = 97
            variability = 1
        
        spo2_values = simulate_ou_process(n_samples, interval, 0.3, base_spo2, variability)
        spo2_values = np.clip(spo2_values, 85, 100)
        
        # Add occasional desaturation events
        if np.random.random() < 0.1:
            desat_idx = np.random.randint(0, len(spo2_values))
            spo2_values[desat_idx] = np.random.randint(88, 93)
        
        spo2_values, quality = add_outliers(spo2_values, persona.outlier_prob, 'spo2')
        
        for i in range(n_samples):
            timestamp = row['start_time'] + timedelta(seconds=i * interval)
            records.append({
                'timestamp': timestamp,
                'spo2_percent': max(0, min(100, spo2_values[i])),
                'context': context,
                'sampling_interval_sec': interval,
                'quality_flag': int(quality[i])
            })
    
    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df[df['timestamp'] >= start_time]
    return df.sort_values('timestamp').reset_index(drop=True) if len(df) > 0 else pd.DataFrame()

def generate_skin_temp_data(
    schedule: pd.DataFrame,
    persona: PersonaConfig,
    start_time: datetime
) -> pd.DataFrame:
    """Generate skin temperature time series."""
    records = []
    
    for _, row in schedule.iterrows():
        context = row['context']
        interval = get_sampling_interval(context, 'skin_temp')
        duration_sec = row['duration_minutes'] * 60
        n_samples = max(1, int(duration_sec / interval))
        
        # Temperature varies by context
        if context == 'sleep':
            base_temp = persona.skin_temp_baseline - 0.5
            variability = 0.2
        elif context in ['active_high', 'active_low', 'post_exercise']:
            base_temp = persona.skin_temp_baseline + 1.0
            variability = 0.4
        else:
            base_temp = persona.skin_temp_baseline
            variability = 0.25
        
        # Menopausal hot flashes
        if persona.name.startswith('Menopausal') and np.random.random() < 0.15:
            base_temp += 1.5
            variability = 0.6
        
        temp_values = simulate_ou_process(n_samples, interval, 0.3, base_temp, variability)
        temp_values = np.clip(temp_values, 32, 40)
        
        temp_values, quality = add_outliers(temp_values, persona.outlier_prob, 'skin_temp')
        
        for i in range(n_samples):
            timestamp = row['start_time'] + timedelta(seconds=i * interval)
            records.append({
                'timestamp': timestamp,
                'skin_temp_celsius': temp_values[i],
                'context': context,
                'sampling_interval_sec': interval,
                'quality_flag': int(quality[i])
            })
    
    df = pd.DataFrame(records)
    df = df[df['timestamp'] >= start_time]
    return df.sort_values('timestamp').reset_index(drop=True)

def generate_motion_data(
    schedule: pd.DataFrame,
    persona: PersonaConfig,
    start_time: datetime
) -> pd.DataFrame:
    """Generate motion/accelerometry data (activity counts)."""
    records = []
    
    for _, row in schedule.iterrows():
        context = row['context']
        interval = get_sampling_interval(context, 'motion')
        duration_sec = row['duration_minutes'] * 60
        n_samples = max(1, int(duration_sec / interval))
        
        # Activity level by context (arbitrary units)
        if context == 'sleep':
            base_motion = 0.05
            variability = 0.02
        elif context == 'active_high':
            base_motion = 2.5
            variability = 0.8
        elif context == 'active_low':
            base_motion = 1.0
            variability = 0.4
        elif context == 'resting_awake':
            base_motion = 0.15
            variability = 0.1
        else:
            base_motion = 0.3
            variability = 0.15
        
        # Motion has very low autocorrelation
        motion_values = np.abs(np.random.normal(base_motion, variability, n_samples))
        
        motion_values, quality = add_outliers(motion_values, persona.outlier_prob, 'motion')
        
        for i in range(n_samples):
            timestamp = row['start_time'] + timedelta(seconds=i * interval)
            records.append({
                'timestamp': timestamp,
                'motion_g': max(0, motion_values[i]),
                'context': context,
                'sampling_interval_sec': interval,
                'quality_flag': int(quality[i])
            })
    
    df = pd.DataFrame(records)
    df = df[df['timestamp'] >= start_time]
    return df.sort_values('timestamp').reset_index(drop=True)

def generate_persona_data(persona_key: str, output_dir: str, days: int = 7):
    """Generate all data files for a single persona."""
    persona = PERSONAS[persona_key]
    start_time = datetime(2024, 11, 25, 0, 0, 0)  # Start at midnight
    
    print(f"Generating data for: {persona.name}")
    print(f"  λ_HR: {persona.lambda_hr}, λ_HRV: {persona.lambda_hrv}, λ_glucose: {persona.lambda_glucose}")
    
    # Generate activity schedule
    schedule = generate_activity_schedule(start_time, days)
    
    # Create persona directory
    persona_dir = os.path.join(output_dir, persona_key)
    os.makedirs(persona_dir, exist_ok=True)
    
    # Save schedule
    schedule.to_csv(os.path.join(persona_dir, 'activity_schedule.csv'), index=False)
    print(f"  - Activity schedule: {len(schedule)} segments")
    
    # Generate each signal type
    hr_df = generate_heart_rate_data(schedule, persona, start_time)
    hr_df.to_csv(os.path.join(persona_dir, 'heart_rate.csv'), index=False)
    print(f"  - Heart rate: {len(hr_df)} samples, {(hr_df['quality_flag'] == 0).sum()} outliers")
    
    hrv_df = generate_hrv_data(schedule, persona, start_time)
    hrv_df.to_csv(os.path.join(persona_dir, 'hrv.csv'), index=False)
    print(f"  - HRV: {len(hrv_df)} samples, {(hrv_df['quality_flag'] == 0).sum()} outliers")
    
    glucose_df = generate_glucose_data(schedule, persona, start_time)
    glucose_df.to_csv(os.path.join(persona_dir, 'glucose.csv'), index=False)
    outliers = glucose_df['quality_flag'] == 0
    print(f"  - Glucose: {len(glucose_df)} samples, {outliers.sum()} outliers")
    
    rr_df = generate_respiratory_rate_data(schedule, persona, start_time)
    rr_df.to_csv(os.path.join(persona_dir, 'respiratory_rate.csv'), index=False)
    print(f"  - Respiratory rate: {len(rr_df)} samples, {(rr_df['quality_flag'] == 0).sum()} outliers")
    
    spo2_df = generate_spo2_data(schedule, persona, start_time)
    if len(spo2_df) > 0:
        spo2_df.to_csv(os.path.join(persona_dir, 'spo2.csv'), index=False)
        print(f"  - SpO2: {len(spo2_df)} samples, {(spo2_df['quality_flag'] == 0).sum()} outliers")
    
    temp_df = generate_skin_temp_data(schedule, persona, start_time)
    temp_df.to_csv(os.path.join(persona_dir, 'skin_temperature.csv'), index=False)
    print(f"  - Skin temp: {len(temp_df)} samples, {(temp_df['quality_flag'] == 0).sum()} outliers")
    
    motion_df = generate_motion_data(schedule, persona, start_time)
    motion_df.to_csv(os.path.join(persona_dir, 'motion.csv'), index=False)
    print(f"  - Motion: {len(motion_df)} samples, {(motion_df['quality_flag'] == 0).sum()} outliers")
    
    # Create metadata file
    metadata = {
        'persona_name': persona.name,
        'persona_key': persona_key,
        'lambda_hr': persona.lambda_hr,
        'lambda_hrv': persona.lambda_hrv,
        'lambda_glucose': persona.lambda_glucose,
        'start_time': str(start_time),
        'days': days,
        'resting_hr_baseline': persona.resting_hr,
        'hrv_rmssd_baseline': persona.hrv_rmssd_baseline,
        'glucose_baseline': persona.glucose_baseline,
        'outlier_probability': persona.outlier_prob
    }
    pd.DataFrame([metadata]).to_csv(os.path.join(persona_dir, 'metadata.csv'), index=False)
    
    print(f"  Data saved to: {persona_dir}/\n")

def main():
    output_dir = '/mnt/user-data/outputs/bioresilience_synthetic_data'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("BioResilienceOS Synthetic Data Generator")
    print("=" * 60)
    print(f"Generating 7 days of Tier 1 (Wearables) and Tier 2 (CGM) data")
    print(f"Output directory: {output_dir}\n")
    
    for persona_key in PERSONAS.keys():
        generate_persona_data(persona_key, output_dir, days=7)
    
    # Create summary file
    summary = []
    for persona_key, persona in PERSONAS.items():
        summary.append({
            'persona_key': persona_key,
            'persona_name': persona.name,
            'lambda_hr': persona.lambda_hr,
            'lambda_hrv': persona.lambda_hrv,
            'lambda_glucose': persona.lambda_glucose,
            'resilience_category': 'High' if persona.lambda_hr >= 0.6 else ('Critical' if persona.lambda_hr <= 0.1 else ('Variable' if 'Menopausal' in persona.name else 'Low'))
        })
    
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, 'personas_summary.csv'), index=False)
    
    print("=" * 60)
    print("Data generation complete!")
    print(f"All files saved to: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
