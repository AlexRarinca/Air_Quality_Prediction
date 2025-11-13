import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/India_complete.csv")

# Parse datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract useful time features
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['dayofweek'] = df['datetime'].dt.dayofweek

# Summary statistics for numeric variables
print(df.describe())

# Check missing values
print(df.isnull().sum())

numeric_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']

# Create subplots for all histograms
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()  # Make indexing easier

for i, col in enumerate(numeric_features):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

df['datetime'] = pd.to_datetime(df['datetime'])  # ensure datetime format

# Extract temporal elements
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df['dayofweek'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
df['weekday_name'] = df['datetime'].dt.day_name()

# Visualization
plt.figure(figsize=(10,5))
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['month'].apply(get_season)

monthly_summary = df.groupby('month')['O3'].mean().reset_index()
sns.barplot(x='month', y='O3', data=monthly_summary)
plt.title('Average Monthly O₃ Levels')
plt.show()

seasonal_summary = df.groupby('season')['O3'].mean().reset_index()
sns.barplot(x='season', y='O3', data=seasonal_summary, order=['Spring', 'Summer', 'Autumn', 'Winter'])
plt.title('Average Seasonal O₃ Levels')
plt.show()


# Remove rows where target variable (O3) is missing
df_clean = df.dropna(subset=['O3'])

# Prepare features and target - EXCLUDE O3 from features since it's our target
feature_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'hour', 'month', 'dayofweek']
X = df_clean[feature_columns]  # Can contain NaN values
y = df_clean['O3']  # Should not contain NaN

# Split into train-test FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Then create and fit model ONLY on training data
model = HistGradientBoostingRegressor(
    random_state=42,
    max_iter=100  # equivalent to n_estimators in RF
)

# Fit directly with NaN values
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f'R²: {r2:.3f}')
print(f'RMSE: {rmse:.3f}')
print(f'MAE: {mae:.3f}')

# Cross-validation - create fresh model for CV
cv_model = HistGradientBoostingRegressor(random_state=42, max_iter=100)
cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='r2')
print(f'Cross-Validation Mean R²: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}')

# Use permutation importance
from sklearn.inspection import permutation_importance

# Calculate permutation importance on test set
perm_importance = permutation_importance(model, X_test, y_test, random_state=42, n_repeats=10)
importances = pd.Series(perm_importance.importances_mean, index=X.columns).sort_values(ascending=False)

print("\nPermutation Feature Importances:")
print(importances)

plt.figure(figsize=(10,6))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Permutation Feature Importance for O₃ Prediction')
plt.xlabel('Importance Score (Decrease in R² when shuffled)')
plt.ylabel('Feature')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual O₃")
plt.ylabel("Predicted O₃")
plt.title("Actual vs Predicted O₃ Levels")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect fit line
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')
plt.show()
