import sys

sys.path.insert(0, ".")

from src.data.load_data import clean_data, create_features, load_raw_data

df = load_raw_data("data/raw/Datos_SIATA_Aire_pm25.json")
df = clean_data(df)
df = create_features(df)
df.to_csv("data/processed/siata_features.csv", index=False)
print("Datos procesados guardados en data/processed/siata_features.csv")