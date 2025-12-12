import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- FILE NAMES ---
FILE_RP5 = "42299.01.01.2018.02.12.2025.1.0.0.en.ansi.00000000.csv"
FILE_API = "open-meteo-27.31N88.59E1636m(1).csv"


def clean_rp5(file_path):
    print(f"Cleaning RP5...")
    try:
        df = pd.read_csv(file_path, sep=';', skiprows=6, encoding='ansi', index_col=False)
    except:
        df = pd.read_csv(file_path, sep=';', skiprows=6, encoding='utf-8', index_col=False)

    # Clean Headers
    df.columns = [c.replace('"', '').strip() for c in df.columns]

    # Rename
    rename_map = {
        'Local time in Gangtok': 'timestamp',
        'T': 'sensor_temp',
        'RRR': 'sensor_rain'
    }
    df = df.rename(columns=rename_map)

    # Parse Time
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M', errors='coerce')

    # Fix Rain
    if 'sensor_rain' in df.columns:
        df['sensor_rain'] = df['sensor_rain'].astype(str).str.replace('Trace of precipitation', '0.05', regex=False)
        df['sensor_rain'] = pd.to_numeric(df['sensor_rain'], errors='coerce').fillna(0.0)

    df = df.dropna(subset=['timestamp'])
    return df.sort_values('timestamp')


def clean_api(file_path):
    print(f"Cleaning Open-Meteo...")
    df = pd.read_csv(file_path, skiprows=3)

    # Rename Columns (Loose match)
    cols = {
        'time': 'timestamp',
        'temperature_2m': 'api_temp',
        'relative_humidity_2m': 'api_humidity',
        'dew_point_2m': 'api_dew_point',
        'surface_pressure': 'api_pressure',
        'wind_speed_10m': 'api_wind_speed',
        'wind_direction_10m': 'api_wind_dir'
    }

    new_cols = {}
    for c in df.columns:
        for k, v in cols.items():
            if k in c:
                new_cols[c] = v

    df = df.rename(columns=new_cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp')


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        df_rp5 = clean_rp5(FILE_RP5)
        df_api = clean_api(FILE_API)

        print("🔗 Merging...")
        df = pd.merge_asof(
            df_rp5, df_api,
            on='timestamp', direction='nearest', tolerance=pd.Timedelta('1h')
        )
        df = df.dropna(subset=['api_temp', 'sensor_temp'])

        # --- FEATURE ENGINEERING (The Science Boost) ---
        print("Adding Physics Features...")

        # 1. Seasonality (Monsoon Awareness)
        df['month'] = df['timestamp'].dt.month
        df['hour'] = df['timestamp'].dt.hour

        # 2. Dew Point Depression (Thermodynamics)
        # If (Temp - DewPoint) is small -> Air is saturated -> RAIN
        if 'api_dew_point' in df.columns:
            df['dew_spread'] = df['api_temp'] - df['api_dew_point']
        else:
            # Approximate if missing
            df['dew_spread'] = df['api_temp'] - ((100 - df['api_humidity']) / 5)

        # 3. Pressure Trend (Storm Warning)
        # Calculate change from 3 hours ago (since RP5 is 3-hourly)
        df['pressure_trend'] = df['api_pressure'].diff().fillna(0)

        # 4. Wind Vectors (Math Fix)
        # 360° and 1° are close, but model thinks they are far. Use Sin/Cos.
        rads = np.deg2rad(df['api_wind_dir'])
        df['wind_sin'] = np.sin(rads)
        df['wind_cos'] = np.cos(rads)

        # --- TRAINING ---
        features = ['api_temp', 'api_humidity', 'api_pressure', 'api_wind_speed',
                    'dew_spread', 'pressure_trend', 'month', 'hour', 'wind_sin', 'wind_cos']

        df['target'] = (df['sensor_rain'] > 0.1).astype(int)

        X = df[features]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Adjust weight slightly less aggressive to fix False Alarms
        # 'sqrt' of the ratio often helps balance Precision/Recall better than raw ratio
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

        model = xgb.XGBClassifier(
            n_estimators=300,  # More trees
            learning_rate=0.03,  # Slower learning = better generalization
            max_depth=6,  # Deeper trees for complex physics
            scale_pos_weight=np.sqrt(ratio)  # Balanced weight
        )

        print("Training V2 Model...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        print(f"\nAccuracy: {accuracy_score(y_test, preds) * 100:.1f}%")
        print(classification_report(y_test, preds))

        model.save_model("sikkim_rain_brain_v2.json")
        print("Saved V2 Model!")

    except Exception as e:
        print(f"Error: {e}")