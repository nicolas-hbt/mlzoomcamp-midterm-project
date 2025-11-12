import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import pickle
import warnings

warnings.filterwarnings('ignore')


def load_data(train_path='train.csv'):
    """
    Loads the raw training data.
    """
    try:
        df = pd.read_csv(train_path)
        return df
    except FileNotFoundError:
        print(f"Error: {train_path} not found.")
        print("Please download the data from: https://www.kaggle.com/competitions/bike-sharing-demand/data")
        return None


def preprocess(df):
    """
    Performs all feature engineering.
    Handles both training (with 'count') and inference (without 'count').
    """
    # --- 1. Create Datetime Features ---
    # The 'datetime' column MUST be present in the input df
    df['datetime'] = pd.to_datetime(df['datetime'])

    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['year'] = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek

    # --- 2. Sin/Cos Cyclical Feature Encoding ---
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31.0)

    # --- 3. Set Categorical Types (Solves Inference Problem) ---
    # This ensures get_dummies creates all columns, even for a single row.
    df['season'] = pd.Categorical(df['season'], categories=[1, 2, 3, 4])
    df['holiday'] = pd.Categorical(df['holiday'], categories=[0, 1])
    df['workingday'] = pd.Categorical(df['workingday'], categories=[0, 1])
    df['weather'] = pd.Categorical(df['weather'], categories=[1, 2, 3, 4])

    # --- 4. One-Hot Encoding for Categoricals ---
    df = pd.get_dummies(df, columns=['season'], prefix='season', drop_first=False, dtype=bool)
    df = pd.get_dummies(df, columns=['holiday'], prefix='holiday', drop_first=False, dtype=bool)
    df = pd.get_dummies(df, columns=['workingday'], prefix='workingday', drop_first=False, dtype=bool)
    df = pd.get_dummies(df, columns=['weather'], prefix='weather', drop_first=False, dtype=bool)

    # --- 5. Define Target (y) and Features (X) ---
    y = None
    if 'count' in df.columns:
        # Only create 'y' if 'count' exists (i.e., during training)
        y = np.log1p(df['count'])

    # Columns to drop (originals, time components, targets)
    columns_to_drop = [
        'datetime', 'hour', 'month', 'day', 'dayofweek',
        'count', 'casual', 'registered'
    ]

    # Drop only the columns that actually exist in the current dataframe
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)

    # 'y' will be None during inference, which is fine
    return X, y


def define_best_model():
    """
    Defines and returns the single best model identified from the analysis notebook.
    """
    print("Defining best model: GB(n_estimators=200, lr=0.1)")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    )
    return model


def run_cross_validation(X, y, model, n_splits=5):
    """
    Runs both Expanding and Sliding Window CV for the single best model.
    """
    # --- 1. Setup CV Splitters and Parameters ---
    tscv_expanding = TimeSeriesSplit(n_splits=n_splits)
    all_splits = list(tscv_expanding.split(X))
    first_fold_train_size = len(all_splits[0][0])

    print(f"--- Initializing CV ---")
    print(f"Total samples: {len(X)}")
    print(f"Number of splits: {n_splits}")
    print(f"Sliding window size set to: {first_fold_train_size} samples")
    print("-" * 30 + "\n")

    results = []
    model_name = "GB(n=200, lr=0.1)"

    # Define our metric: RMSE. Because y is log-transformed, this *is* RMSLE.
    rmse = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

    # --- 2. Run Evaluation Loop ---

    # === A. Expanding Window ===
    print("Running Expanding Window CV...")
    for fold, (train_index, test_index) in enumerate(all_splits, 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"  Expanding Fold {fold}/{n_splits}...")

        # We clone the model to ensure it's fresh for each fold
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        preds_log = model_clone.predict(X_test)
        score = rmse(y_test, preds_log)  # y_test is already log(y)

        results.append({
            "cv_method": "Expanding",
            "model": model_name,
            "fold": fold,
            "rmsle": score
        })

    # === B. Sliding Window ===
    print("\nRunning Sliding Window CV...")
    for fold, (train_index, test_index) in enumerate(all_splits, 1):
        sliding_train_index = train_index[-first_fold_train_size:]
        X_train, X_test = X.iloc[sliding_train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[sliding_train_index], y.iloc[test_index]

        print(f"  Sliding Fold {fold}/{n_splits}...")

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        preds_log = model_clone.predict(X_test)
        score = rmse(y_test, preds_log)

        results.append({
            "cv_method": "Sliding",
            "model": model_name,
            "fold": fold,
            "rmsle": score
        })

    print("\n--- Evaluation Complete ---")
    return pd.DataFrame(results)


def summarize_results(results_df):
    """
    Creates and prints the final pivot table of mean scores.
    """
    final_scores = results_df.pivot_table(
        index="model",
        columns="cv_method",
        values="rmsle",
        aggfunc="mean"
    )

    final_scores['mean_rmsle'] = final_scores.mean(axis=1)
    final_scores = final_scores.sort_values(by='mean_rmsle')

    print("\n=== Final Model Comparison (Mean RMSLE) ===")
    print(final_scores.to_markdown(floatfmt=".6f"))


def train_and_save_model(X, y, model, filename="gb_model.pkl"):
    """
    Trains the final model on ALL data and saves it to a pickle file.
    """
    print(f"\nTraining final model on {len(X)} samples...")

    # We use the original model object, which is unfitted
    model.fit(X, y)

    print(f"Saving model to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")


def main():
    """
    Main function to run the entire pipeline.
    """
    print("Loading data...")
    # Assumes 'train.csv' is in the same directory
    df = load_data('train.csv')

    if df is not None:
        print("Preprocessing data and engineering features...")
        X, y = preprocess(df)

        print("Defining model...")
        best_model = define_best_model()

        print("Starting cross-validation...")
        # Pass the single model object
        results_df = run_cross_validation(X, y, best_model, n_splits=5)

        print("Summarizing results...")
        summarize_results(results_df)

        print("Training and saving final model...")
        # Pass the *original* unfitted model to be retrained on all data
        train_and_save_model(X, y, best_model, filename="gb_model.pkl")


if __name__ == "__main__":
    main()