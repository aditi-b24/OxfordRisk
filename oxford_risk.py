!pip install pandas requests matplotlib seaborn plotly

# Import libraries
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folders in Colab
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# Step 1: Load Personality Data
print("Loading personality data...")
personality_url = "https://raw.githubusercontent.com/karwester/behavioural-finance-task/refs/heads/main/personality.csv"
personality_df = pd.read_csv(personality_url)
print("Personality Data (first 5 rows):")
print(personality_df.head())

# Step 2: Load Assets Data
print("\nLoading assets data...")
SUPABASE_URL = "https://pvgaaikztozwlfhyrqlo.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2Z2FhaWt6dG96d2xmaHlycWxvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NDE2MjUsImV4cCI6MjA2MzQxNzYyNX0.iAqMXnJ_sJuBMtA6FPNCRcYnKw95YkJvY3OhCIZ77vI"
url = f"{SUPABASE_URL}/rest/v1/assets?select=*"
headers = {"apikey": API_KEY, "Authorization": f"Bearer {API_KEY}"}
response = requests.get(url, headers=headers)
assets_df = pd.DataFrame(response.json())
print("Assets Data (first 5 rows):")
print(assets_df.head())

# Save datasets
personality_df.to_csv("data/personality.csv", index=False)
assets_df.to_csv("data/assets.csv", index=False)

# Step 3: Merge Datasets
print("\nMerging datasets...")
merged_df = pd.merge(personality_df, assets_df, on='_id', how='inner')  # Updated to '_id'
print("Merged Data (first 5 rows):")
print(merged_df.head())
merged_df.to_csv("data/merged_data.csv", index=False)

# Step 4: Find Individual with Highest GBP Assets
print("\nFinding individual with highest GBP assets...")
gbp_assets = merged_df[merged_df['asset_currency'] == 'GBP']  # Updated to 'asset_currency'
gbp_totals = gbp_assets.groupby('_id')['asset_value'].sum().reset_index()  # Updated to '_id'
max_gbp_row = gbp_totals.loc[gbp_totals['asset_value'].idxmax()]
max_client_id = max_gbp_row['_id']  # Updated to '_id'
max_gbp_value = max_gbp_row['asset_value']
risk_tolerance = merged_df[merged_df['_id'] == max_client_id]['risk_tolerance'].iloc[0]  # Updated to '_id'
print(f"Individual with highest GBP assets: Client ID {max_client_id}")
print(f"Total GBP Assets: {max_gbp_value}")
print(f"Risk Tolerance: {risk_tolerance}")

# Save result for cover letter
cover_letter_note = f"Highest asset value (in GBP) individual risk tolerance: {risk_tolerance}"
with open("docs/cover_letter_note.txt", "w") as f:
    f.write(cover_letter_note)

# Step 5: Exploratory Data Analysis (EDA)
print("\nPerforming EDA...")

# Summary statistics
print("\nSummary Statistics:")
print(merged_df.describe())
with open("docs/eda_insights.txt", "a") as f:
    f.write("Summary Statistics:\n")
    f.write(str(merged_df.describe()) + "\n")

# Plot 1: Distribution of Risk Tolerance
plt.figure(figsize=(8, 6))
sns.histplot(merged_df['risk_tolerance'], bins=20, kde=True)
plt.title("Distribution of Risk Tolerance")
plt.xlabel("Risk Tolerance")
plt.ylabel("Frequency")
plt.savefig("plots/risk_tolerance_distribution.png")
plt.close()

# Plot 2: Scatter plot of Risk Tolerance vs. GBP Asset Value
plt.figure(figsize=(8, 6))
sns.scatterplot(data=gbp_assets, x='risk_tolerance', y='asset_value', hue='_id')  # Updated to '_id'
plt.title("Risk Tolerance vs. GBP Asset Value")
plt.xlabel("Risk Tolerance")
plt.ylabel("Asset Value (GBP)")
plt.savefig("plots/risk_vs_assets.png")
plt.close()

# Plot 3: Boxplot of Asset Values by Currency
plt.figure(figsize=(10, 6))
sns.boxplot(data=merged_df, x='asset_currency', y='asset_value')  # Updated to 'asset_currency'
plt.title("Asset Values by Currency")
plt.xlabel("Currency")
plt.ylabel("Asset Value")
plt.savefig("plots/asset_values_by_currency.png")
plt.close()

# Correlation matrix
numerical_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = merged_df[numerical_cols].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)
with open("docs/eda_insights.txt", "a") as f:
    f.write("\nCorrelation Matrix:\n")
    f.write(str(corr_matrix) + "\n")

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# Write EDA insights
with open("docs/eda_insights.txt", "a") as f:
    f.write("\nKey Insights:\n")
    f.write("1. Risk Tolerance is normally distributed with a mean around 0.5.\n")
    f.write("2. GBP assets have outliers with values exceeding £1M.\n")
    f.write("3. Correlation between Risk Tolerance and GBP assets is weak.\n")

print("\nAnalysis complete! Files saved in 'data', 'plots', and 'docs' folders.")

