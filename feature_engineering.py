import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_processed_data(ev_path=r"data\preprocessed\ev_battery_health_preprocessed.csv",
                        traffic_path=r"data\preprocessed\traffic_preprocessed.csv",
                        weather_path=r"data\preprocessed\weatherHistory_preprocessed.csv"):
    ev = pd.read_csv(ev_path)
    traffic = pd.read_csv(traffic_path)
    weather = pd.read_csv(weather_path)
    return ev, traffic, weather

def create_datetime_features(df, datetime_col):
    """Extract hour, day, month, and weekday from datetime."""
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df['hour'] = df[datetime_col].dt.hour
        df['day'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        df['weekday'] = df[datetime_col].dt.weekday
    return df

def create_ev_features(ev):
    if 'total_distance_km' in ev.columns and 'trip_duration_min' in ev.columns:
        ev['calc_avg_speed_kmph'] = ev['total_distance_km'] / (ev['trip_duration_min'] / 60)
    
    if 'charging_cycles' in ev.columns and 'fast_charging_ratio_%' in ev.columns:
        ev['fast_charging_stress'] = ev['charging_cycles'] * (ev['fast_charging_ratio_%'] / 100)
    
    if 'average_battery_temperature_C' in ev.columns and 'ambient_temperature_C' in ev.columns:
        ev['thermal_stress'] = ev['average_battery_temperature_C'] - ev['ambient_temperature_C']
    if 'battery_health_%' in ev.columns and 'total_distance_km' in ev.columns:
        ev['degradation_per_km'] = (100 - ev['battery_health_%']) / ev['total_distance_km'].replace(0, np.nan)
        ev['degradation_per_1000km'] = ev['degradation_per_km'] * 1000
    
    return ev

def encode_categorical(df):
    """Encode categorical variables."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def handle_missing_values(df):
    """Fill missing numeric values with median."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def save_featured_data(df, path):
    df.to_csv(path, index=False)
    print(f"Feature-engineered dataset saved to {path}")

def main():
    ev, traffic, weather = load_processed_data()
    ev = create_ev_features(ev)
    ev = encode_categorical(ev)
    ev = handle_missing_values(ev)
    save_featured_data(ev, "ev_featured.csv")
    traffic = create_datetime_features(traffic, 'timestamp')
    traffic = encode_categorical(traffic)
    traffic = handle_missing_values(traffic)
    save_featured_data(traffic, "traffic_featured.csv")
    weather = create_datetime_features(weather, 'Formatted Date')
    weather = encode_categorical(weather)
    weather = handle_missing_values(weather)
    save_featured_data(weather, "weather_featured.csv")

def save_featured_data(df, path):
    df.to_csv(path, index=False)
    print(f"\nFeature-engineered dataset saved to {path}")
    print(f"Final Features: {list(df.columns)}")


if __name__ == "__main__":
    main()
