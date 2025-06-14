import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Parameters
states = ['Telangana', 'Delhi', 'Andhra Pradesh', 'Karnataka']
diseases = [
    'Fever', 'Malaria', 'Diarrhea', 'Cold', 'Cough', 'Headache',
    'Eye Infection', 'Skin Rash', 'Dengue', 'Allergy'
]
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
years = list(range(2016, 2025))
entries_per_combo = 60  # number of entries per (State, Disease, Year, Season)

# Season sinusoidal effect (higher in some seasons)
season_multiplier = {
    'Winter': 1.2,
    'Spring': 0.9,
    'Summer': 1.1,
    'Autumn': 1.0
}

# Generate synthetic data
data = []
for state in states:
    for disease in diseases:
        base_incidence = random.randint(5, 30)
        for year in years:
            for season in seasons:
                for _ in range(entries_per_combo):
                    # Generate a date within the season
                    month = {'Winter': 1, 'Spring': 4, 'Summer': 7, 'Autumn': 10}[season]
                    day = random.randint(1, 28)
                    date = datetime(year, month, day)

                    # Sine wave based seasonal fluctuation + noise
                    seasonal_effect = season_multiplier[season]
                    noise = np.random.normal(loc=0.0, scale=3.0)
                    count = int(base_incidence * seasonal_effect + noise)
                    count = max(count, 0)  # no negative counts

                    data.append({
                        'Date': date.strftime("%Y-%m-%d"),
                        'State': state,
                        'Disease': disease,
                        'Disease_Count': count,
                        'Year': year,
                        'Month': month,
                        'Season': season
                    })

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/synthetic_disease_data.csv", index=False)

print(" synthetic_disease_data.csv saved successfully.")
print(f" Total rows: {len(df)}")
