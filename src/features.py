"""Shared, deterministic feature preparation for training and inference."""
import numpy as np
import pandas as pd

def make_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    hour = df['date'].dt.hour
    dow = df['date'].dt.day_of_week
    df['month'] = df['date'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    return df.drop(columns=['date', 'rv1', 'rv2'], errors='ignore')
