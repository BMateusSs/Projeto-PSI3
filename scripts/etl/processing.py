import pandas as pd

red_wine_path = "data/raw/winequality-red.csv"
white_wine_path = "data/raw/winequality-white.csv"

red_wine = pd.read_csv(red_wine_path, sep=",")

white_wine = pd.read_csv(white_wine_path, sep=";", quotechar='"')

red_wine["wine_type"] = "red"
white_wine["wine_type"] = "white"

wine_quality = pd.concat([red_wine, white_wine], ignore_index=True)

path = 'data/raw/combined_wine_quality.csv'
wine_quality.to_csv(path, sep=';', index=False)
