import pandas as pd

df = pd.read_csv("data/raw/combined_wine_quality.csv")

df.to_parquet("data/processed/wine-quality.parquet", engine="pyarrow")