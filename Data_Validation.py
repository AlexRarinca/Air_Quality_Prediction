import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt # Added for plotting

# Classification Models to Test
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def load_data(file_path, sheet_name=0):
    """Loads data from an Excel file."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def run_ml_validation(data_path, features, target_col, sheet_name=0, n_splits=5):
    """
    Reads data, preprocesses it, and runs cross-validation for a suite of ML models.

    Args:
        data_path (str): Path to the Excel file.
        features (list): List of column names to use as features (X).
        target_col (str): Name of the column to use as the target (y).
        sheet_name (int/str): Name or index of the Excel sheet.
        n_splits (int): Number of folds for K-Fold cross-validation.
    """

    df = load_data(data_path, sheet_name)
    if df is None:
        return

    # Drop rows where the target column is NaN
    df = df.dropna(subset=[target_col])

    print(f"--- Loaded data with {len(df)} rows and {len(df.columns)} columns. ---")

    # 1. Prepare Features (X) and Target (y)
    try:
        X = df[features]
        y = df[target_col]
    except KeyError as e:
        print(f"Error: One or more specified columns not found: {e}")
        return

    # 2. Identify Column Types for Preprocessing
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    print(f"Numerical Features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")

    # 3. Create Preprocessing Pipelines
    # Numerical pipeline: Impute missing values with the median, then scale.
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing values with the most frequent, then one-hot encode.
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Ignore categories not seen during training
    ])

    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough' # Keep any other columns (e.g., IDs, if not explicitly removed)
    )

    
   # 4. Define Models to Test
    models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # 5. Run Cross-Validation and Report Results
    results = {}
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("\n--- Starting Model Validation (5-Fold Cross-Validation, Metric: R^2 Score) ---")

    for name, model in models.items():
        # Create a final pipeline: Preprocessor -> Model
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])


    # Calculate cross-validation scores (using 'r2' as the default metric for regression)
    scores = cross_val_score(full_pipeline, X, y, cv=cv, scoring='r2', n_jobs=-1)

    mean_score = scores.mean()
    std_dev = scores.std()

    results[name] = {'Mean R^2 Score': mean_score, 'Std Dev': std_dev}

    print(f"\nModel: {name}")
    print(f"  Validation Scores: {scores}")
    print(f"  Mean R^2 Score: {mean_score:.4f} (±{std_dev:.4f})")

    print("\n==================================================")
    print("      Summary of Model Validation Results         ")
    print("==================================================")

    results_df = pd.DataFrame(results).T.sort_values(by='Mean R^2 Score', ascending=False)
    print(results_df.to_markdown(floatfmt=".4f"))
    print("\nInterpretation:")
    print("The models with the highest 'Mean R^2 Score' and lowest 'Std Dev' are the best initial candidates for this dataset and task.")

    # 6. Plotting Results (New Section)
    print("\n--- Generating Visualization ---")

    # Prepare data for plotting
    r2_scores = results_df['Mean R^2 Score']
    std_dev = results_df['Std Dev']
    model_names = results_df.index

    plt.figure(figsize=(12, 6))
    # Create a bar chart with error bars
    plt.bar(model_names, r2_scores, yerr=std_dev, capsize=5, color='teal', edgecolor='black', alpha=0.8)
    plt.title('ML Model Validation Results (Cross-Validated R^2 Score)')
    plt.xlabel('Machine Learning Model')
    plt.ylabel('Mean R^2 Score')
    y_min = min(0.0, r2_scores.min() - 0.1)
    y_max = max(1.0, r2_scores.max() + 0.1)
    plt.ylim(y_min, y_max)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show() # Display the plot


if __name__ == '__main__':
    # --- CONFIGURATION ---
    # 1. Update this to the name of your Excel file.
    EXCEL_FILE = 'C:\\Users\jalia\Documents\Year 3 Eng Maths\Scientific Computing\Python\MDM3\station_daytest.xlsx'
    
    # 2. Update this to the name or index of the sheet containing your data.
    SHEET_NAME = 'station_day'  
    
    # 3. **MANDATORY:** Update this to the name of the column you want to predict (e.g., 'Target_Variable', 'Success_Flag', 'Price').
    TARGET_COLUMN = 'O3' 

    # 4. **MANDATORY:** Update this list with ALL the column names you want to use as features (inputs) for prediction.
    FEATURE_COLUMNS = [
        
        'Date'
        'StationId' 
        'PM2.5'	
        'PM10'	
        'NO'	
        'NO2'	
        'NOx'	
        'NH3'
        'CO'
        'SO2'
    ]


    run_ml_validation(
        data_path=EXCEL_FILE,
        sheet_name=SHEET_NAME,
        features=FEATURE_COLUMNS,
        target_col=TARGET_COLUMN
    )

