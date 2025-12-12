import pandas as pd
import math
import datetime
import firebase_admin
from firebase_admin import credentials, db

# ---------------------------------------------------------------------
#                          FIREBASE SETUP
# ---------------------------------------------------------------------
cred = credentials.Certificate("agrosmart-f6758-firebase-adminsdk-fbsvc-ead2ed827d.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://agrosmart-f6758-default-rtdb.firebaseio.com/"
})

FARM_ROOT = "Niranj/FarmData/Node1"   # Adjusted exactly as per uploaded JSON


# ---------------------------------------------------------------------
#                         IRRIGATION MODEL
# ---------------------------------------------------------------------
SOIL_FC = 32.0
SOIL_WP = 16.0
LATITUDE = 26.9


class MaizeSmartIrrigation:
    def __init__(self, csv_path):
        self.schedule = self._load_schedule(csv_path)

    def _load_schedule(self, path):
        df = pd.read_csv(path)
        processed = []

        for _, row in df.iterrows():
            d_start, d_end = map(float, str(row["Period_Days"]).split("-"))

            t_str = str(row["Temp_Range_C"]).replace("°C", "")
            t_max_ref = float(t_str.split("-")[1])

            m_str = str(row["Moisture_Target_Range"]).replace("%", "")
            m_min, m_max = map(float, m_str.split("-"))
            base_target = (m_min + m_max) / 2

            processed.append({
                "start": d_start, "end": d_end,
                "stage": row["Stage"],
                "theta_base": base_target,
                "temp_max_ref": t_max_ref,
                "root_depth": float(row["Root_Depth_mm"])
            })

        return pd.DataFrame(processed)

    def estimate_solar_radiation(self, day_of_year, t_max, t_min):
        lat_rad = math.radians(LATITUDE)
        dr = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
        declination = 0.409 * math.sin((2 * math.pi * day_of_year / 365) - 1.39)
        ws = math.acos(-math.tan(lat_rad) * math.tan(declination))

        ra = (24 * 60 / math.pi) * 0.0820 * dr * (
            ws * math.sin(lat_rad) * math.sin(declination) +
            math.cos(lat_rad) * math.cos(declination) * math.sin(ws)
        )

        t_mean = (t_max + t_min) / 2
        et0 = 0.0023 * ra * (t_mean + 17.8) * math.sqrt(t_max - t_min)

        return round(et0, 2)

    def calculate_3hr_update(self, day_after_sowing, t_current, t_max_forecast, t_min_forecast, moisture_current):
        stage_data = self.schedule[
            (self.schedule["start"] <= day_after_sowing) &
            (self.schedule["end"] > day_after_sowing)
        ].iloc[0]

        target = stage_data["theta_base"]

        instant_overshoot = max(0, t_current - stage_data["temp_max_ref"])

        day_of_year = datetime.datetime.now().timetuple().tm_yday
        est_et0 = self.estimate_solar_radiation(day_of_year, t_max_forecast, t_min_forecast)

        et_buffer = 5.0 if est_et0 > 5.5 else 0.0

        final_target = target + (instant_overshoot * 2.0) + et_buffer
        final_target = min(90.0, final_target)

        if moisture_current < final_target:
            deficit_pct = final_target - moisture_current
            aw_fraction = (SOIL_FC - SOIL_WP) / 100.0
            water_mm = (deficit_pct / 100.0) * aw_fraction * stage_data["root_depth"]
        else:
            water_mm = 0

        return {
            "Stage": stage_data["stage"],
            "Target_Moisture": round(final_target, 1),
            "Current_Moisture": moisture_current,
            "Water_Required_mm": round(water_mm, 2)
        }


# ---------------------------------------------------------------------
#                   FIREBASE READ / WRITE FUNCTIONS
# ---------------------------------------------------------------------

def read_node1_data():
    """Reads soil data from Niranj/FarmData/Node1."""
    ref = db.reference(FARM_ROOT)
    data = ref.get()

    if data is None:
        raise Exception("Firebase path does not exist.")

    soil_temp = float(data.get("SoilTemperature", 0))     # Exists in Node1
    soil_moist = float(data.get("SoilMoisture", 0))       # Exists in Node1

    return soil_temp, soil_moist


def write_expected_moisture(expected_value):
    """Write the expected moisture % to Firebase."""
    ref = db.reference(f"{FARM_ROOT}/expectedWater")
    ref.set(expected_value)
    print(f"Wrote expectedWater = {expected_value}")


# ---------------------------------------------------------------------
#                              MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    soil_temp, soil_moist = read_node1_data()
    print("Soil Temp:", soil_temp)
    print("Soil Moisture:", soil_moist)

    irrigation = MaizeSmartIrrigation("maize_data.csv")

    result = irrigation.calculate_3hr_update(
        day_after_sowing=115,
        t_current=soil_temp,
        t_max_forecast=35,
        t_min_forecast=22,
        moisture_current=soil_moist
    )

    expected = result["Target_Moisture"]
    write_expected_moisture(expected)

    print("Irrigation Result:", result)
