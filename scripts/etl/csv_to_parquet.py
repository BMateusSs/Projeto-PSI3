import pandas as pd

df = pd.read_csv("data/raw/winemag-data-130k-v2.csv")

df.to_parquet("winemag.parquet", engine="pyarrow")