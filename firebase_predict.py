import requests
import pandas as pd
import xgboost as xgb
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

# ---------------------------------------------------------------------
#                       FIREBASE INITIALIZATION
# ---------------------------------------------------------------------
cred = credentials.Certificate("agrosmart-f6758-firebase-adminsdk-fbsvc-ead2ed827d.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://agrosmart-f6758-default-rtdb.firebaseio.com/"
})

WEATHER_PATH = "Niranj/WeatherData"   # Confirmed from uploaded DB


# ---------------------------------------------------------------------
#                       MODEL + API PARAMETERS
# ---------------------------------------------------------------------
MODEL_FILE = "sikkim_fusion_brain.json"
LAT = 27.31
LON = 88.59

# Constant pressure (your requirement)
FIXED_PRESSURE = 1020.0


# ---------------------------------------------------------------------
#                     READ WEATHER DATA FROM FIREBASE
# ---------------------------------------------------------------------
def read_weather_from_firebase():
    ref = db.reference(WEATHER_PATH)
    data = ref.get()

    # Based on your JSON upload:
    # Temperature: 23.45
    # Humidity: 64
    # RainPercent: 43
    # WindSpeed: 89
    # :contentReference[oaicite:1]{index=1}

    temp = float(data.get("Temperature", 0))
    humidity = float(data.get("Humidity", 0))
    wind_speed = float(data.get("WindSpeed", 0))

    return {
        "sensor_temp": temp,
        "sensor_humidity": humidity,
        "sensor_pressure": FIXED_PRESSURE,
        "sensor_wind_speed": wind_speed
    }


# ---------------------------------------------------------------------
#                     WRITE RAIN PERCENT BACK TO FIREBASE
# ---------------------------------------------------------------------
def write_rain_percent_to_firebase(rain_percent):
    ref = db.reference(f"{WEATHER_PATH}/RainPercent")
    ref.set(rain_percent)
    print(f"Updated Firebase → RainPercent = {rain_percent}")


# ---------------------------------------------------------------------
#                       API LIVE METEOROLOGY
# ---------------------------------------------------------------------
def get_live_forecast():
    print("Fetching live forecast from Open-Meteo...")
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,relative_humidity_2m,surface_pressure,"
        "wind_speed_10m,wind_direction_10m,dew_point_2m&current_weather=true"
    )

    try:
        r = requests.get(url).json()
        current_hour_iso = datetime.now().strftime("%Y-%m-%dT%H:00")
        hourly_times = r['hourly']['time']

        try:
            idx = hourly_times.index(current_hour_iso)
        except ValueError:
            idx = 0

        return {
            'api_temp': r['hourly']['temperature_2m'][idx],
            'api_humidity': r['hourly']['relative_humidity_2m'][idx],
            'api_pressure': r['hourly']['surface_pressure'][idx],
            'api_wind_speed': r['hourly']['wind_speed_10m'][idx],
            'api_wind_dir': r['hourly']['wind_direction_10m'][idx],
            'api_dew_point': r['hourly']['dew_point_2m'][idx]
        }

    except Exception as e:
        print(f"API ERROR: {e}")
        return None


# ---------------------------------------------------------------------
#                                 MAIN
# ---------------------------------------------------------------------
def main():
    print(f"\nSIKKIM RAIN PREDICTOR – {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Load ML model
    try:
        model = xgb.Booster()
        model.load_model(MODEL_FILE)
    except Exception as e:
        print(f"Model load error: {e}")
        return

    # Get live forecast
    api_data = get_live_forecast()
    if not api_data:
        return

    # Get weather data from Firebase (instead of local sensors)
    sensor_data = read_weather_from_firebase()

    current_time = datetime.now()
    rads = np.deg2rad(api_data['api_wind_dir'])

    # Derived physics variables
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
        print(f"Feature missing: {e}")
        return

    dmatrix = xgb.DMatrix(input_df)
    probability = model.predict(dmatrix)[0]
    rain_percentage = int(probability * 100)

    print("\n========= FORECAST =========")
    print(f"API: {api_data['api_temp']}°C / {api_data['api_humidity']}% RH")
    print(f"Firebase Sensor: {sensor_data['sensor_temp']}°C / {sensor_data['sensor_humidity']}% RH")
    print(f"Dew Spread: {sensor_dew_spread:.1f}")
    print(f"Rain chance: {rain_percentage}%")
    print("============================\n")

    # Upload result to Firebase
    write_rain_percent_to_firebase(rain_percentage)


if __name__ == "__main__":
    main()
