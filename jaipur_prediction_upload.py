import requests
import pandas as pd
import xgboost as xgb
import numpy as np
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db

MODEL_FILE = "Jaipur_fusion_brain.json"
LAT = 26.8439
LON = 75.5652

SERVICE_ACCOUNT_FILE = "agrosmart-f6758-firebase-adminsdk-fbsvc-ead2ed827d.json"
DATABASE_URL = "https://agrosmart-f6758-default-rtdb.firebaseio.com"
FARM_ID = "Kisan1"


def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred, {
            "databaseURL": DATABASE_URL
        })


def upload_rain_percent_to_firebase(percentage):
    try:
        init_firebase()
        ref_path = f"{FARM_ID}/WeatherData"
        ref = db.reference(ref_path)

        # Only update RainPercent, keep the rest as-is
        ref.update({
            "RainPercent": int(percentage)
        })

        print(f"[FIREBASE] Updated {ref_path}/RainPercent = {int(percentage)}")
    except Exception as e:
        print(f"[FIREBASE ERROR] {e}")


def get_live_forecast():
    print("Fetching live forecast from Open-Meteo...")
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,"
        f"wind_speed_10m,wind_direction_10m,dew_point_2m&current_weather=true"
    )

    try:
        r = requests.get(url).json()
        current_hour_iso = datetime.now().strftime("%Y-%m-%dT%H:00")
        hourly_times = r['hourly']['time']

        try:
            idx = hourly_times.index(current_hour_iso)
        except ValueError:
            idx = 0

        data = {
            'api_temp': r['hourly']['temperature_2m'][idx],
            'api_humidity': r['hourly']['relative_humidity_2m'][idx],
            'api_pressure': r['hourly']['surface_pressure'][idx],
            'api_wind_speed': r['hourly']['wind_speed_10m'][idx],
            'api_wind_dir': r['hourly']['wind_direction_10m'][idx],
            'api_dew_point': r['hourly']['dew_point_2m'][idx]
        }
        return data
    except Exception as e:
        print(f"API Error: {e}")
        return None


def read_sensors():
    print("Reading local sensors...")
    return {
        'sensor_temp': 12,
        'sensor_humidity': 88.0,
        'sensor_pressure': 1019.0,
        'sensor_wind_speed': 5
    }


def main():
    print(f"\nSIKKIM RAIN PREDICTOR - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    try:
        model = xgb.Booster()
        model.load_model(MODEL_FILE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    api_data = get_live_forecast()
    if not api_data:
        return

    sensor_data = read_sensors()
    current_time = datetime.now()

    rads = np.deg2rad(api_data['api_wind_dir'])

    api_dew_spread = api_data['api_temp'] - api_data['api_dew_point']
    sensor_dew_spread = sensor_data['sensor_temp'] - ((100 - sensor_data['sensor_humidity']) / 5)

    pressure_diff = api_data['api_pressure'] - sensor_data['sensor_pressure']
    temp_diff = api_data['api_temp'] - sensor_data['sensor_temp']

    pressure_trend = 0.0

    input_dict = {
        'api_temp': api_data['api_temp'],
        'api_humidity': api_data['api_humidity'],
        'api_pressure': api_data['api_pressure'],
        'api_wind_speed': api_data['api_wind_speed'],
        'api_dew_spread': api_dew_spread,
        'pressure_trend': pressure_trend,
        'wind_sin': np.sin(rads),
        'wind_cos': np.cos(rads),
        'sensor_temp': sensor_data['sensor_temp'],
        'sensor_humidity': sensor_data['sensor_humidity'],
        'sensor_pressure': sensor_data['sensor_pressure'],
        'sensor_wind_speed': sensor_data['sensor_wind_speed'],
        'sensor_dew_spread': sensor_dew_spread,
        'month': current_time.month,
        'hour': current_time.hour,
        'pressure_diff': pressure_diff,
        'temp_diff': temp_diff
    }

    feature_order = [
        'api_temp', 'api_humidity', 'api_pressure', 'api_wind_speed',
        'api_dew_spread', 'pressure_trend', 'wind_sin', 'wind_cos',
        'sensor_temp', 'sensor_humidity', 'sensor_pressure', 'sensor_wind_speed',
        'sensor_dew_spread',
        'month', 'hour', 'pressure_diff', 'temp_diff'
    ]

    try:
        input_df = pd.DataFrame([input_dict])[feature_order]
    except KeyError as e:
        print(f"CRITICAL ERROR: Missing feature in input dictionary: {e}")
        return

    dmatrix = xgb.DMatrix(input_df)
    probability = model.predict(dmatrix)[0]
    percentage = int(probability * 100)

    print("\n" + "=" * 30)
    print(f"FORECAST ANALYSIS")
    print(f"   API reading: {api_data['api_temp']}°C, {api_data['api_humidity']}% RH")
    print(f"   Sensor reading:  {sensor_data['sensor_temp']}°C, {sensor_data['sensor_humidity']}% RH")
    print(f"   Physics Check:   Dew Spread={sensor_dew_spread:.1f}, P-Diff={pressure_diff:.1f}")
    print("=" * 30)

    if percentage > 50:
        print(f"\nRAIN ALERT: {percentage}% Probability")
    else:
        print(f"\nNO RAIN: {percentage}% Probability")
    print("=" * 30)
    upload_rain_percent_to_firebase(percentage)


if __name__ == "__main__":
    main()
