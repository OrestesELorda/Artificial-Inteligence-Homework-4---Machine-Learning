# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

NUMERIC_FEATURES = ["price", "cost", "units_sold", "promotion_frequency", "shelf_level", "profit"]
CATEGORICAL_FEATURES = ["product_name", "category", "product_id"]

def preprocess_dataframe(df):
    """
    1. Report missing values
    2. Drop rows with too many missing values (>2)
    3. Impute numeric with median, categorical with mode
    4. Outlier detection (IQR) and capping
    5. Scaling: StandardScaler (z-score)
    Returns:
        df_processed (pandas DataFrame),
        summary (dict) for display
    """
    summary = {}
    # missing values
    missing = df.isna().sum()
    summary['missing_per_column'] = missing.to_dict()

    # drop rows with too many missing values (>=3)
    df = df.copy()
    df['n_missing_row'] = df.isna().sum(axis=1)
    dropped_rows = int((df['n_missing_row'] >= 3).sum())
    df = df[df['n_missing_row'] < 3].drop(columns=['n_missing_row'])
    summary['dropped_rows_due_to_missing'] = dropped_rows

    # impute numeric with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            if df[col].isna().sum() > 0:
                median = df[col].median()
                df[col] = df[col].fillna(median)
                summary[f'imputed_{col}'] = float(median)

    # impute categorical with mode
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].isna().sum() > 0:
            mode = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode)
            summary[f'imputed_{col}'] = str(mode)

    # outlier detection and capping using IQR for numeric features
    cap_info = {}
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        # store original extremes
        original_min = df[col].min()
        original_max = df[col].max()
        # cap
        df[col] = df[col].clip(lower=lower, upper=upper)
        cap_info[col] = {
            'lower_cap': float(lower),
            'upper_cap': float(upper),
            'original_min': float(original_min),
            'original_max': float(original_max)
        }
    summary['outlier_capping'] = cap_info

    # scaling: use StandardScaler (z-score) because k-means is distance-based
    scaler = StandardScaler()
    numeric_cols_present = [c for c in NUMERIC_FEATURES if c in df.columns]
    df_scaled = df.copy()
    df_scaled[numeric_cols_present] = scaler.fit_transform(df_scaled[numeric_cols_present])
    summary['scaling'] = 'StandardScaler (z-score) applied to numeric features'

    return df_scaled, summary