import pandas as pd
import math
import datetime

SOIL_FC = 32.0
SOIL_WP = 16.0
LATITUDE = 26.9


class MaizeSmartIrrigation:
    def __init__(self, csv_path):
        self.schedule = self._load_schedule(csv_path)

    def _load_schedule(self, path):
        """Loads and processes the CSV into a lookup table."""
        try:
            df = pd.read_csv(path)
            processed = []
            for _, row in df.iterrows():
                # Parse '0-14' into start/end
                d_start, d_end = map(float, str(row['Period_Days']).split('-'))

                # Parse '10-18°C' into Max Reference Temp
                t_str = str(row['Temp_Range_C']).replace('°C', '')
                t_max_ref = float(t_str.split('-')[1])

                # Parse '50-60%' into Base Target (Average)
                m_str = str(row['Moisture_Target_Range']).replace('%', '')
                m_min, m_max = map(float, m_str.split('-'))
                base_target = (m_min + m_max) / 2

                processed.append({
                    'start': d_start, 'end': d_end,
                    'stage': row['Stage'],
                    'theta_base': base_target,
                    'temp_max_ref': t_max_ref,
                    'root_depth': float(row['Root_Depth_mm'])
                })
            return pd.DataFrame(processed)
        except Exception as e:
            print(f"CSV Load Error: {e}")
            return pd.DataFrame()

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
        """
        The Main Decision Loop (Runs every 3 hours).
        """

        stage_data = self.schedule[
            (self.schedule['start'] <= day_after_sowing) &
            (self.schedule['end'] > day_after_sowing)
            ].iloc[0]

        target = stage_data['theta_base']


        instant_overshoot = max(0, t_current - stage_data['temp_max_ref'])

        day_of_year = datetime.datetime.now().timetuple().tm_yday
        estimated_et0 = self.estimate_solar_radiation(day_of_year, t_max_forecast, t_min_forecast)

        et_buffer = 5.0 if estimated_et0 > 5.5 else 0.0

        final_target = target + (instant_overshoot * 2.0) + et_buffer
        final_target = min(90.0, final_target)  # Cap at 90% (leave 10% for air)

        if moisture_current < final_target:
            deficit_pct = final_target - moisture_current

            available_water_fraction = (SOIL_FC - SOIL_WP) / 100.0
            water_mm = (deficit_pct / 100.0) * available_water_fraction * stage_data['root_depth']
        else:
            water_mm = 0

        return {
            "Stage": stage_data['stage'],
            "Root_Depth": stage_data['root_depth'],
            "Condition": "Heat Stress" if instant_overshoot > 0 else "Normal",
            "Solar_Demand_ET0": estimated_et0,
            "Target_Moisture": round(final_target, 1),
            "Current_Moisture": moisture_current,
            "Water_Required_mm": round(water_mm, 2)
        }


# --- SIMULATION (Example of one 3-hour check) ---
system = MaizeSmartIrrigation('maize_data.csv')

# Scenario: 2:00 PM, Hot Day (34C), Day 115 (Silking)
result = system.calculate_3hr_update(
    day_after_sowing=115,
    t_current=30.0,
    t_max_forecast=35.0,
    t_min_forecast=22.0,
    moisture_current=70.0
)

print(f"--- 3-HOUR IRRIGATION CHECK ---")
print(f"Stage: {result['Stage']} (Root Depth: {result['Root_Depth']}mm)")
print(f"Est. ET0 (Solar Demand): {result['Solar_Demand_ET0']} mm/day")
print(f"Thermal Status: {result['Condition']}")
print(f"Target Moisture: {result['Target_Moisture']}% (Adjusted for Heat)")
print(f"Current Moisture: {result['Current_Moisture']}%")