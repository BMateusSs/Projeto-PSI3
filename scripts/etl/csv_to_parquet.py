import pandas as pd

df = pd.read_csv("data/raw/combined_wine_quality.csv", sep=';')

df.to_parquet("data/processed/wine-quality.parquet", engine="pyarrow", index=False)
