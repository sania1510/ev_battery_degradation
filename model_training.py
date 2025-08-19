import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Utility helpers
# -----------------------------



def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def plot_pred_vs_actual(y_true, y_pred, title, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_residuals(y_true, y_pred, title, out_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=40, kde=True)
    plt.title(title)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(model, feature_names, title, out_path):
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(7, max(3, 0.3*len(importances))))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


# -----------------------------
# Training blocks (no merging)
# -----------------------------

def train_ev_model(ev_path: str):
    """Train baseline and advanced models on EV dataset to predict degradation or battery_health_%.
    Uses the following priority for target:
      1) 'degradation_per_1000km' if present
      2) 'degradation_per_km' if present
      3) 100 - battery_health_% (computed)
    """
    df = pd.read_csv(ev_path)

    # Pick target
    target_col = None
    for cand in ["degradation_per_1000km", "degradation_per_km"]:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        if "battery_health_%" in df.columns:
            df["computed_degradation"] = 100 - df["battery_health_%"].astype(float)
            target_col = "computed_degradation"
        else:
            raise ValueError("EV dataset must contain either degradation_per_1000km, degradation_per_km, or battery_health_%.")

    # Features: drop obvious non-features
    drop_cols = {target_col}
    # Never use raw battery_health_% as a feature when predicting degradation proxy
    drop_cols.update([c for c in ["battery_health_%"] if c in df.columns])

    X = df.drop(columns=list(drop_cols))
    y = df[target_col].astype(float)

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number]).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42)
    }

    ev_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        ev_results[name] = metrics

        # Save model
        joblib.dump(model, f"models/EV_{name}.pkl")

        # Plots
        plot_pred_vs_actual(y_test, preds, f"EV - {name} Predicted vs Actual", f"results/EV_{name}_pred_vs_actual.png")
        plot_residuals(y_test, preds, f"EV - {name} Residuals", f"results/EV_{name}_residuals.png")
        plot_feature_importance(model, X.columns, f"EV - {name} Feature Importance", f"results/EV_{name}_feature_importance.png")

    with open("results/EV_metrics.json", "w") as f:
        json.dump(ev_results, f, indent=4)

    print("\n[EV] Finished training. Metrics:")
    print(json.dumps(ev_results, indent=2))


def train_traffic_model(traffic_path: str):
    """Train models on traffic dataset to predict Vehicles (traffic intensity).
    This gives a learned \"Traffic Stress\" proxy that you can use for scenario analysis.
    """
    df = pd.read_csv(traffic_path)

    # Basic cleaning / types
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df["hour"] = df["DateTime"].dt.hour
        df["weekday"] = df["DateTime"].dt.weekday
        df["month"] = df["DateTime"].dt.month

    target_col = "Vehicles"
    drop_cols = [target_col, "DateTime", "ID"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col].astype(float)

    # Encode Junction if still non-numeric
    if "Junction" in X.columns and not np.issubdtype(X["Junction"].dtype, np.number):
        X["Junction"] = X["Junction"].astype("category").cat.codes

    X = X.select_dtypes(include=[np.number]).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42)
    }

    traffic_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        traffic_results[name] = metrics

        joblib.dump(model, f"models/Traffic_{name}.pkl")
        plot_pred_vs_actual(y_test, preds, f"Traffic - {name} Predicted vs Actual", f"results/Traffic_{name}_pred_vs_actual.png")
        plot_residuals(y_test, preds, f"Traffic - {name} Residuals", f"results/Traffic_{name}_residuals.png")
        plot_feature_importance(model, X.columns, f"Traffic - {name} Feature Importance", f"results/Traffic_{name}_feature_importance.png")

    with open("results/Traffic_metrics.json", "w") as f:
        json.dump(traffic_results, f, indent=4)

    print("\n[Traffic] Finished training. Metrics:")
    print(json.dumps(traffic_results, indent=2))


def train_weather_model(weather_path: str):
    """Train models on weather dataset to predict Temperature (C) as a stability/complexity check
    and to derive a \"Weather Stress\" proxy for scenario analysis.
    """
    df = pd.read_csv(weather_path)

    # Fix common typo/variant in cloud cover column
    if "Loud Cover" in df.columns and "Cloud Cover" not in df.columns:
        df.rename(columns={"Loud Cover": "Cloud Cover"}, inplace=True)

    # Datetime engineering if present
    if "Formatted Date" in df.columns:
        df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], errors="coerce")
        if "hour" not in df.columns:
            df["hour"] = df["Formatted Date"].dt.hour
        if "day" not in df.columns:
            df["day"] = df["Formatted Date"].dt.day
        if "month" not in df.columns:
            df["month"] = df["Formatted Date"].dt.month
        if "weekday" not in df.columns:
            df["weekday"] = df["Formatted Date"].dt.weekday

    target_col = "Temperature (C)" if "Temperature (C)" in df.columns else None
    if target_col is None:
        raise ValueError("Weather dataset must contain 'Temperature (C)' as target for this training block.")

    drop_cols = [target_col, "Formatted Date", "Daily Summary", "Summary", "Precip Type"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col].astype(float)

    # Encode any remaining non-numeric features safely
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = X[c].astype("category").cat.codes

    X = X.select_dtypes(include=[np.number]).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42)
    }

    weather_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        weather_results[name] = metrics

        joblib.dump(model, f"models/Weather_{name}.pkl")
        plot_pred_vs_actual(y_test, preds, f"Weather - {name} Predicted vs Actual", f"results/Weather_{name}_pred_vs_actual.png")
        plot_residuals(y_test, preds, f"Weather - {name} Residuals", f"results/Weather_{name}_residuals.png")
        plot_feature_importance(model, X.columns, f"Weather - {name} Feature Importance", f"results/Weather_{name}_feature_importance.png")

    with open("results/Weather_metrics.json", "w") as f:
        json.dump(weather_results, f, indent=4)

    print("\n[Weather] Finished training. Metrics:")
    print(json.dumps(weather_results, indent=2))


# -----------------------------
# Scenario analysis helper (optional, no merging)
# -----------------------------

def degradation_scenario(ev_usage_row: dict, traffic_row: dict, weather_row: dict) -> dict:
    """Estimate *relative* degradation risk using trained models independently.
    This does NOT merge datasets; it uses each trained model to produce a comparable score
    and combines them heuristically for interpretability. The output is a normalized
    risk index to compare scenarios like 'high traffic + cloudy' vs 'low traffic + clear'.
    """
    # Load models if present
    models = {}
    for key in ["EV_LinearRegression", "Traffic_RandomForest", "Weather_RandomForest"]:
        p = f"models/{key}.pkl"
        if os.path.exists(p):
            models[key] = joblib.load(p)

    scores = {}

    # EV usage baseline score (predicted degradation proxy)
    if "EV_LinearRegression" in models:
        ev_df = pd.DataFrame([ev_usage_row])
        ev_df = ev_df.select_dtypes(include=[np.number])
        scores["ev_baseline"] = float(models["EV_LinearRegression"].predict(ev_df)[0])

    # Traffic stress score ~ predicted vehicles
    if "Traffic_RandomForest" in models:
        tf_df = pd.DataFrame([traffic_row])
        # encode Junction if present and not numeric
        if "Junction" in tf_df.columns and not np.issubdtype(tf_df["Junction"].dtype, np.number):
            tf_df["Junction"] = tf_df["Junction"].astype("category").cat.codes
        tf_df = tf_df.select_dtypes(include=[np.number])
        scores["traffic_stress"] = float(models["Traffic_RandomForest"].predict(tf_df)[0])

    # Weather stress score ~ predicted temperature (proxy) or use direct numeric features
    if "Weather_RandomForest" in models:
        w_df = pd.DataFrame([weather_row])
        for c in w_df.columns:
            if not np.issubdtype(w_df[c].dtype, np.number):
                w_df[c] = w_df[c].astype("category").cat.codes
        w_df = w_df.select_dtypes(include=[np.number])
        scores["weather_stress"] = float(models["Weather_RandomForest"].predict(w_df)[0])

    # Normalize and combine into a relative risk index
    # (purely heuristic since datasets are not merged)
    vals = [v for v in scores.values() if np.isfinite(v)]
    if len(vals) == 0:
        return {"message": "No models available to compute scenario."}

    mean_v, std_v = (np.mean(vals), np.std(vals) if np.std(vals) > 0 else 1.0)
    norm_scores = {k: (v - mean_v) / std_v for k, v in scores.items()}

    # Weighted sum (tuneable weights)
    combined = (
        0.6 * norm_scores.get("ev_baseline", 0.0) +
        0.25 * norm_scores.get("traffic_stress", 0.0) +
        0.15 * norm_scores.get("weather_stress", 0.0)
    )

    norm_scores["combined_risk_index"] = float(combined)
    return norm_scores


# -----------------------------
# Entrypoint to run all trainings
# -----------------------------
if __name__ == "__main__":
    ensure_dirs()

    # Update these paths if your filenames differ
    EV_PATH = r"data\preprocessed\ev_battery_health_preprocessed.csv"
    TRAFFIC_PATH = r"E:\EV battery classifaication forcasting\data\preprocessed\traffic_preprocessed.csv"
    WEATHER_PATH = r"data\preprocessed\weatherHistory_preprocessed.csv"

    train_ev_model(EV_PATH)
    train_traffic_model(TRAFFIC_PATH)
    train_weather_model(WEATHER_PATH)

    print("\nAll training completed. Models saved in ./models and reports in ./results")
