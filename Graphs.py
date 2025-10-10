import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load all datasets
India_complete = pd.read_csv(r'C:\Users\jalia\Documents\Year 3 Eng Maths\Scientific Computing\Python\India_complete.csv')

# Display basic information about the datasets
# print(f"\nDataset Shapes:")
# print(f"India_complete_day: {India_complete.shape}")

# print("=" * 50)
# print("CITY_DAY DATASET")
# print("=" * 50)
# print("\nFirst few rows:")
# print(India_complete.head())
# print("\nData Info:")
# print(India_complete.info())
# print("\nBasic Statistics:")
# print(India_complete.describe())
# print("\nMissing Values:")
# print(India_complete.isnull().sum())

# Data Preprocessing
df = India_complete.copy()
# Convert Date to datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract time features
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Day'] = df['datetime'].dt.day
df['DayOfWeek'] = df['datetime'].dt.dayofweek
df['Quarter'] = df['datetime'].dt.quarter
df['Hour'] = df['datetime'].dt.hour


#fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# # 03 distribution
# axes[0].hist(df['AQI'].dropna(), bins=50, color='blue', edgecolor='black')
# axes[0].set_title('Distribution of AQI Values', fontsize=14, fontweight='bold')
# axes[0].set_xlabel('AQI', fontsize=12)
# axes[0].set_ylabel('Frequency', fontsize=12)
# axes[0].axvline(df['AQI'].mean(), color='red', linestyle='--', label=f'Mean: {df["AQI"].mean():.2f}')
# axes[0].legend()

# # AQI Bucket distribution
# aqi_bucket_counts = df['AQI_Bucket'].value_counts()
# axes[1].bar(aqi_bucket_counts.index, aqi_bucket_counts.values, color='red', edgecolor='black')
# axes[1].set_title('Distribution of AQI Buckets', fontsize=14, fontweight='bold')
# axes[1].set_xlabel('AQI Bucket', fontsize=12)
# axes[1].set_ylabel('Count', fontsize=12)
# axes[1].tick_params(axis='x', rotation=45)
# plt.tight_layout()
# plt.show()




plt.figure(figsize=(10, 6))

# Plot the histogram of O3 values
plt.hist(df['O3'].dropna(), bins=50, color='blue', edgecolor='black')

# Add titles and labels
plt.title('Distribution of O3 Values', fontsize=14, fontweight='bold')
plt.xlabel('O3', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Add a vertical line for the mean of the O3 column
plt.axvline(df['O3'].mean(), color='red', linestyle='--', label=f'Mean: {df["O3"].mean():.2f}')
plt.legend()

# Display the plot
plt.show()


# Print O3 statistics

print(f"\nO3 Statistics:")
print(f"Mean O3: {df['O3'].mean():.2f}")
print(f"Median O3: {df['O3'].median():.2f}")
print(f"Min O3: {df['O3'].min():.2f}")
print(f"Max O3: {df['O3'].max():.2f}")




pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2']

# Remove pollutants with too many missing values
available_pollutants = [p for p in pollutants if p in df.columns and df[p].notna().sum() > 100]

fig, axes = plt.subplots(2, 4, figsize=(18, 12))
axes = axes.flatten()

for idx, pollutant in enumerate(available_pollutants):
   

    if idx < len(axes):
        axes[idx].hist(df[pollutant].dropna(), bins=30, color='purple', edgecolor='black')
        axes[idx].set_title(f'{pollutant} Distribution', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(pollutant, fontsize=9)
        axes[idx].set_ylabel('Frequency', fontsize=9)
        
    

# Hide unused subplots
for idx in range(len(available_pollutants), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()


# Select numeric columns for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Air Quality Parameters', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Show strong correlations with 03
if 'O3' in correlation_matrix.columns:
    O3_corr = correlation_matrix['O3'].sort_values(ascending=False)
    print("\nCorrelation with O3:")
    print(O3_corr[1:11])


# # Year-over-Year O3 Trend
# yearly_O3 = df.groupby('Year')['O3'].mean().reset_index()

# plt.figure(figsize=(12, 6))
# plt.plot(yearly_O3['Year'], yearly_O3['O3'], marker='o', linewidth=2, 
#          markersize=8, color='darkred')
# plt.title('Year-over-Year O3 Trend', fontsize=16, fontweight='bold')
# plt.xlabel('Year', fontsize=12)
# plt.ylabel('Average O3', fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.xticks(yearly_O3['Year'])
# plt.tight_layout()
# plt.show()

# print("\nYearly O3 Summary:")
# print(yearly_O3)

# Month-over-Month O3 Trend
monthly_O3 = df.groupby('Month')['O3'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(monthly_O3['Month'], monthly_O3['O3'], marker='o', linewidth=2, 
         markersize=8, color='darkred')
plt.title('Month-over-Month O3 Trend', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average O3', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(monthly_O3['Month'])
plt.tight_layout()
plt.show()

print("\nMonthly O3 Summary:")
print(monthly_O3)

# Day of the week O3 Trend
day_of_week_O3 = df.groupby('DayOfWeek')['O3'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(day_of_week_O3['DayOfWeek'], day_of_week_O3['O3'], marker='o', linewidth=2, 
         markersize=8, color='lightgreen')
plt.title('Day-Of-The-Week O3 Trend', fontsize=16, fontweight='bold')
plt.xlabel('DayOfWeek', fontsize=12)
plt.ylabel('Average O3', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(day_of_week_O3['DayOfWeek'])
plt.tight_layout()
plt.show()

print("\nDaily O3 Summary:")
print(day_of_week_O3)


# Get top 50 days with Highest O3
top_O3_days = df.nlargest(50, 'O3')[['Day', 'Month','Year', 'City', 'AQI', 'PM2.5', 'PM10', 'AQI_Bucket', 'O3', 'NO2', 'SO2', 'CO']]

print("=" * 50)
print("Top 50 days with Highest O3")
print("=" * 50)
print(top_O3_days.to_string(index=False))

# Visualize top polluted cities
plt.figure(figsize=(12, 6))
top_polluted_cities = top_O3_days['City'].value_counts()
plt.bar(top_polluted_cities.index, top_polluted_cities.values, 
        color='darkred', edgecolor='black', alpha=0.7)
plt.title('Cities with Most High-O3 Days (Top 50)', fontsize=16, fontweight='bold')
plt.xlabel('City', fontsize=12)
plt.ylabel('Number of High-O3 Days', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Stations by average O3
Stations_O3 = df.groupby('StationName')['O3'].mean().sort_values(ascending=False)
Stations_O3.plot(kind='barh', ax=axes[0], color='orange', edgecolor='black')
axes[0].set_title('Station by Average O3', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Average O3', fontsize=12)
axes[0].set_ylabel('Station', fontsize=12)
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=4)
# Station Id by average O3
StationId_O3 = df.groupby('StationId')['O3'].mean().sort_values(ascending=False)
StationId_O3.plot(kind='barh', ax=axes[1], color='orange', edgecolor='black')
axes[1].set_title('Station ID by Average O3', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Average O3', fontsize=12)
axes[1].set_ylabel('Station ID', fontsize=12)
axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=4)


plt.tight_layout()
plt.show()



# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Cities by average O3
city_O3 = df.groupby('City')['O3'].mean().sort_values(ascending=False)
city_O3.plot(kind='barh', ax=axes[0], color='orange', edgecolor='black')
axes[0].set_title('Cities by Average O3', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Average O3', fontsize=12)
axes[0].set_ylabel('City', fontsize=12)


# State by average O3
State_O3 = df.groupby('State')['O3'].mean().sort_values(ascending=False)
State_O3.plot(kind='barh', ax=axes[1], color='orange', edgecolor='black')
axes[1].set_title('State by Average O3', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Average O3', fontsize=12)
axes[1].set_ylabel('State', fontsize=12)



plt.tight_layout()

plt.show()
