import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

FILE_RP5 = "42348.01.01.2018.08.12.2025.1.0.0.en.ansi.00000000.csv"


def clean_rp5(file_path):
    print(f"Cleaning RP5 (Sensor Data)...")
    try:
        df = pd.read_csv(file_path, sep=';', skiprows=6, encoding='ansi', index_col=False)
    except:
        df = pd.read_csv(file_path, sep=';', skiprows=6, encoding='utf-8', index_col=False)

    df.columns = [c.replace('"', '').strip() for c in df.columns]

    rename_map = {
        'Local time in Jaipur / Sanganer (airport)': 'timestamp',
        'T': 'sensor_temp',
        'U': 'sensor_humidity',
        'RRR': 'sensor_rain',
        'Ff': 'sensor_wind_speed',
        'Po': 'sensor_pressure'
    }
    df = df.rename(columns=rename_map)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M', errors='coerce')

    # Fix Numeric Columns
    if 'sensor_rain' in df.columns:
        df['sensor_rain'] = df['sensor_rain'].astype(str).str.replace('Trace of precipitation', '0.05', regex=False)
        df['sensor_rain'] = pd.to_numeric(df['sensor_rain'], errors='coerce').fillna(0.0)

    for col in ['sensor_temp', 'sensor_humidity', 'sensor_pressure', 'sensor_wind_speed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['timestamp'])
    return df.sort_values('timestamp')

if __name__ == "__main__":
    try:
        df = clean_rp5(FILE_RP5)

        df['future_rain'] = df['sensor_rain'].shift(-1)

        df = df.dropna(subset=['future_rain'])

        print("Adding Sensor-Only Physics...")

        df['month'] = df['timestamp'].dt.month
        df['hour'] = df['timestamp'].dt.hour

        df['pressure_trend'] = df['sensor_pressure'].diff().fillna(0)

        df['dew_spread'] = df['sensor_temp'] - ((100 - df['sensor_humidity']) / 5)

        df['storm_index'] = (df['pressure_trend'] * -1) * df['sensor_wind_speed']

        features = [
            'sensor_temp', 'sensor_humidity', 'sensor_pressure', 'sensor_wind_speed',
            'month', 'hour', 'pressure_trend', 'dew_spread', 'storm_index'
        ]

        df['target'] = (df['future_rain'] > 0.1).astype(int)

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

        print("Training Standalone Sensor Model...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        print(f"\nAccuracy (Predicting Next 3 Hours): {accuracy_score(y_test, preds) * 100:.1f}%")
        print(classification_report(y_test, preds))

        model.save_model("jaipur_sensor_only_brain.json")
        print("Saved 'jaipur_sensor_only_brain.json'")

    except Exception as e:
        print(f"Error: {e}")