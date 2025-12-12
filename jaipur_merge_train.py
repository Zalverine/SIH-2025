import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

FILE_RP5 = "42348.01.01.2018.08.12.2025.1.0.0.en.ansi.00000000.csv"
FILE_API = "open-meteo-26.82N75.55E380m(1).csv"


def clean_rp5(file_path):
    print(f"Cleaning RP5 (Ground Truth)...")
    try:
        df = pd.read_csv(file_path, sep=';', skiprows=6, encoding='ansi', index_col=False)
    except:
        df = pd.read_csv(file_path, sep=';', skiprows=6, encoding='utf-8', index_col=False)

    df.columns = [c.replace('"', '').strip() for c in df.columns]

    rename_map = {
        'Local time in Jaipur / Sanganer (airport)': 'timestamp',
        'T': 'sensor_temp',
        'RRR': 'sensor_rain',
        'U': 'sensor_humidity',
        'Po': 'sensor_pressure',
        'Ff': 'sensor_wind_speed'
    }
    df = df.rename(columns=rename_map)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M', errors='coerce')

    if 'sensor_rain' in df.columns:
        df['sensor_rain'] = df['sensor_rain'].astype(str).str.replace('Trace of precipitation', '0.05', regex=False)
        df['sensor_rain'] = pd.to_numeric(df['sensor_rain'], errors='coerce').fillna(0.0)

    cols_to_fix = ['sensor_temp', 'sensor_humidity', 'sensor_pressure', 'sensor_wind_speed']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['timestamp'])
    return df.sort_values('timestamp')


def clean_api(file_path):
    print(f"Cleaning Open-Meteo...")
    df = pd.read_csv(file_path, skiprows=3)

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

if __name__ == "__main__":
    try:
        df_rp5 = clean_rp5(FILE_RP5)
        df_api = clean_api(FILE_API)

        print("Merging...")
        df = pd.merge_asof(
            df_rp5, df_api,
            on='timestamp', direction='nearest', tolerance=pd.Timedelta('1h')
        )

        df = df.dropna(subset=['api_temp', 'sensor_temp', 'sensor_pressure', 'sensor_humidity'])

        print("Adding Sensor Fusion Features...")

        df['month'] = df['timestamp'].dt.month
        df['hour'] = df['timestamp'].dt.hour

        rads = np.deg2rad(df['api_wind_dir'])
        df['wind_sin'] = np.sin(rads)
        df['wind_cos'] = np.cos(rads)

        if 'api_dew_point' in df.columns:
            df['api_dew_spread'] = df['api_temp'] - df['api_dew_point']
        else:
            df['api_dew_spread'] = df['api_temp'] - ((100 - df['api_humidity']) / 5)

        df['sensor_dew_spread'] = df['sensor_temp'] - ((100 - df['sensor_humidity']) / 5)
        df['pressure_diff'] = df['api_pressure'] - df['sensor_pressure']
        df['temp_diff'] = df['api_temp'] - df['sensor_temp']
        df['pressure_trend'] = df['api_pressure'].diff().fillna(0)

        features = [

            'api_temp', 'api_humidity', 'api_pressure', 'api_wind_speed',
            'api_dew_spread', 'pressure_trend', 'wind_sin', 'wind_cos',
            'sensor_temp', 'sensor_humidity', 'sensor_pressure', 'sensor_wind_speed',
            'sensor_dew_spread',
            'month', 'hour', 'pressure_diff', 'temp_diff'
        ]

        # Target
        df['target'] = (df['sensor_rain'] > 0.1).astype(int)

        X = df[features]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            scale_pos_weight=np.sqrt(ratio)
        )

        print("Training Sensor Fusion Model...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        print(f"\nAccuracy: {accuracy_score(y_test, preds) * 100:.1f}%")
        print(classification_report(y_test, preds))

        model.save_model("Jaipur_fusion_brain.json")
        print("Saved 'Jaipur_fusion_brain.json'!")

    except Exception as e:
        print(f"Error: {e}")