import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("./dataset/synthetic_disease_data.csv")

# 1. Required columns check
required_columns = {'Date', 'State', 'Disease', 'Disease_Count', 'Year', 'Month', 'Season'}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"⛔ Missing required columns: {missing}")
else:
    print("✅ All required columns are present.")

# 2. Year distribution
year_counts = df['Year'].value_counts().sort_index()
print(f"\n📅 Year Range: {df['Year'].min()} - {df['Year'].max()}")
print("📊 Entries per year:\n", year_counts)

# 3. Check (State, Disease) pairs with >= 4 years
pair_years = df.groupby(['State', 'Disease'])['Year'].nunique().reset_index(name='YearCount')
valid_pairs = pair_years[pair_years['YearCount'] >= 4]
print(f"\n✅ (State, Disease) pairs with ≥ 4 years of data: {len(valid_pairs)}")
print(valid_pairs.head(10))

# 4. Disease count stats
print("\n📈 Disease_Count stats:")
print(df['Disease_Count'].describe())

# 5. Optional visualization
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Season', y='Disease_Count')
plt.title(" Disease Count by Season")
plt.grid(True)
plt.savefig("visualisation/disease_count_by_season.png")
plt.close()

# Optional: heatmap of valid pair distributions
pivot = valid_pairs.pivot(index='State', columns='Disease', values='YearCount').fillna(0)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title(" State-Disease Pairs with Year Coverage")
plt.tight_layout()
plt.savefig("visualisation/state_disease_year_coverage.png")
plt.close()

print("\n Visuals saved to 'visualisation/' folder.")
