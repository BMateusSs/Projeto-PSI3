import numpy as np
import pandas as pd

def load_data():
    return pd.read_parquet("data/raw/winemag.parquet")

def remove_columns(df):
    df = df.drop(columns=['region_2', 'taster_twitter_handle'])
    return df.dropna(subset=['variety'])

def fill_missing_values(df):
    df['country'] = df['country'].fillna('Unknown')
    df['designation'] = df['designation'].fillna('Non-designated')
    df['taster_name'] = df['taster_name'].fillna('Anonymous')
    return df

def safe_mode(series):
    modes = series.mode()
    return modes[0] if not modes.empty else 'Unknown'

def fill_province(df):
    province_mode = df.groupby('country')['province'].transform(safe_mode)
    df['province'] = df['province'].fillna(province_mode)
    return df

def fill_price(df):
    price_median = df.groupby(['variety', 'country'])['price'].transform('median')
    df['price'] = df['price'].fillna(price_median)
    return df

def fill_region(df):
    df['region_1'] = np.where(
        df['region_1'].isna(),
        df['province'] + " (Unspecified)",
        df['region_1']
    )
    return df

def save_data(df, path):
    df.to_parquet(path)

def process_data():
    df = load_data()
    df = remove_columns(df)
    df = fill_missing_values(df)
    df = fill_province(df)
    df = fill_price(df)
    df = fill_region(df)
    save_data(df, 'data/processed/winemag.parquet')
    return df

if __name__ == "__main__":
    process_data()