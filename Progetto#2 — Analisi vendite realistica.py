import pandas as pd
import numpy as np

np.random.seed(42)

# =====================================================
# PARTE 1 – CREA I DATASET
# =====================================================

# 1. Ordini.csv: 100.000 righe con ClienteID, ProdottoID, Quantità e DataOrdine.
n_ordini = 100_000
date_range = pd.date_range("2025-01-01", "2025-12-31", freq="D")

ordini = pd.DataFrame({
    "ClienteID": np.random.randint(1, 5001, size=n_ordini, dtype="int32"),
    "ProdottoID": np.random.randint(1, 21, size=n_ordini, dtype="int16"),
    "Quantita": np.random.randint(1, 6, size=n_ordini, dtype="int8"),
    "DataOrdine": np.random.choice(date_range, size=n_ordini)
})
ordini.to_csv("ordini.csv", index=False)

# 2. prodotti.json: 20 prodotti con Categoria e Fornitore.
prodotti = pd.DataFrame({
    "ProdottoID": np.arange(1, 21, dtype="int16"),
    "Categoria": np.random.choice(["Tech", "Casa", "Sport", "Giochi"], size=20),
    "Fornitore": np.random.choice(["Amazon", "VendorX", "VendorY"], size=20),
    "Prezzo": np.round(np.random.uniform(5, 200, size=20), 2)
})
prodotti.to_json("prodotti.json", orient="records", indent=2)

# 3. clienti.csv: 5.000 clienti con Regione e Segmento.
clienti = pd.DataFrame({
    "ClienteID": np.arange(1, 5001, dtype="int32"),
    "Regione": np.random.choice(["Nord", "Centro", "Sud", "Isole"], size=5000),
    "Segmento": np.random.choice(["Consumer", "Business", "VIP"], size=5000)
})
clienti.to_csv("clienti.csv", index=False)

# =====================================================
# PARTE 2 – CREARE UN DATAFRAME UNIFICATO
# =====================================================

# 4. Unisci ordini.  (leggo ordini.csv)
ordini_df = pd.read_csv("ordini.csv", parse_dates=["DataOrdine"])

# 5. Unisci prodotti. (join con prodotti.json)
prodotti_df = pd.read_json("prodotti.json")
df = ordini_df.merge(prodotti_df, on="ProdottoID", how="left")

# 6. Unisci clienti. (join con clienti.csv)
clienti_df = pd.read_csv("clienti.csv")
df = df.merge(clienti_df, on="ClienteID", how="left")

# =====================================================
# PARTE 3 – OTTIMIZZAZIONE
# =====================================================

# 7. Ottimizzare i tipi di dato.
df["ClienteID"] = df["ClienteID"].astype("int32")
df["ProdottoID"] = df["ProdottoID"].astype("int16")
df["Quantita"] = df["Quantita"].astype("int8")
df["Prezzo"] = df["Prezzo"].astype("float32")
for col in ["Categoria", "Fornitore", "Regione", "Segmento"]:
    df[col] = df[col].astype("category")

# 8. Ottimizzare l’uso della memoria. (stampo prima/dopo)
print("Memoria dopo ottimizzazione:")
print(df.info(memory_usage="deep"))

# =====================================================
# PARTE 4 – CREARE COLONNE E FILTRARE I DATI
# =====================================================

# 9. Crea colonna ValoreTotale = Prezzo * Quantità.
df["ValoreTotale"] = df["Prezzo"] * df["Quantita"]

# 10. Filtrare ordini con ValoreTotale > 100 e clienti (filtro sugli ordini).
df_filtrato = df[df["ValoreTotale"] > 100]

print("\nPrime righe DataFrame unificato:")
print(df.head())

print("\nPrime righe ordini con ValoreTotale > 100:")
print(df_filtrato.head())
