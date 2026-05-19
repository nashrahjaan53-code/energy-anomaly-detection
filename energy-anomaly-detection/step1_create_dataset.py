import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
os.makedirs('data', exist_ok=True)
def generate_energy_data(n_days=365):
    np.random.seed(42)
    
  
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=x) for x in range(n_days * 24)]
    
    data = []
    for ts in timestamps:
        hour = ts.hour
        base_usage = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
        day_of_year = ts.timetuple().tm_yday
        temp = 20 + 10 * np.sin((day_of_year / 365) * 2 * np.pi) + np.random.normal(0, 2)
        temp_effect = 0.1 * abs(temp - 20)
        is_weekend = 1 if ts.weekday() >= 5 else 0
        weekend_effect = 0.2 if is_weekend else 0
        usage = base_usage + temp_effect + weekend_effect + np.random.normal(0, 0.05)
        
        # Label: 0=Normal, 1=Anomaly
        anomaly = 0
        if np.random.random() < 0.02:
            anomaly = 1
            usage *= np.random.uniform(2.5, 4.0) if np.random.random() > 0.3 else np.random.uniform(0.1, 0.3)

        data.append([ts, usage, temp, is_weekend, anomaly])

    df = pd.DataFrame(data, columns=['timestamp', 'usage_kwh', 'temperature', 'is_weekend', 'anomaly'])
    df.to_csv('data/energy_data.csv', index=False)
    print(f"✅ Dataset created with {len(df)} rows.")

if __name__ == "__main__":
    generate_energy_data()