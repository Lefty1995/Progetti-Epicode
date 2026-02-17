import pandas as pd
import numpy as np

# =====================================================
# CREAZIONE DATASET FITTIZIO VENDITE
# =====================================================

np.random.seed(42)

date_range = pd.date_range("2025-01-01", "2025-03-31", freq="D")
prodotti = ["A001", "A002", "B010"]

rows = []
for d in date_range:
    for p in prodotti:
        # introduciamo qualche NaN e valori strani apposta
        vendite = np.random.poisson(lam=20)
        if np.random.rand() < 0.05:
            vendite = np.nan            # valore mancante
        prezzo = np.random.choice([10.0, 10.5, 9.9, np.nan])
        rows.append([d, p, vendite, prezzo])

df = pd.DataFrame(rows, columns=["Data", "Prodotto", "Vendite", "Prezzo"])

# aggiungo duplicati artificiali
df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)

# =====================================================
# PARTE 1 - CARICAMENTO ED ESPLORAZIONE
# =====================================================

print("Prime righe:")
print(df.head(), "\n")

print("Info struttura:")
print(df.info(), "\n")      # tipi di dato

print("Statistiche descrittive:")
print(df.describe(), "\n")  # solo colonne numeriche

# =====================================================
# PARTE 2 - PULIZIA
# =====================================================

# 3. Gestione valori mancanti
#   - per Vendite: sostituisco con 0 (nessuna vendita registrata)
#   - per Prezzo: sostituisco con la media del prodotto
df["Vendite"] = df["Vendite"].fillna(0)

df["Prezzo"] = df.groupby("Prodotto")["Prezzo"] \
                .transform(lambda s: s.fillna(s.mean()))

# 4. Rimozione duplicati (stessa Data, Prodotto, Vendite, Prezzo)
df = df.drop_duplicates()

# 5. Controllo / correzione tipi
df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
df["Vendite"] = df["Vendite"].astype("int32")
df["Prezzo"] = df["Prezzo"].astype("float32")

print("Dopo pulizia:")
print(df.info(), "\n")

# =====================================================
# PARTE 3 - ANALISI ESPLORATIVA
# =====================================================

# 6. Vendite totali per prodotto
vendite_totali_prodotto = df.groupby("Prodotto")["Vendite"].sum()
print("Vendite totali per prodotto:")
print(vendite_totali_prodotto, "\n")

# 7. Prodotto più venduto e meno venduto
prodotto_piu_venduto = vendite_totali_prodotto.idxmax()
prodotto_meno_venduto = vendite_totali_prodotto.idxmin()

print(f"Prodotto più venduto: {prodotto_piu_venduto}")
print(f"Prodotto meno venduto: {prodotto_meno_venduto}\n")

# 8. Vendite medie giornaliere (somma su tutti i prodotti / numero giorni)
vendite_giornaliere = df.groupby("Data")["Vendite"].sum()
vendite_medie_giornaliere = vendite_giornaliere.mean()

print("Vendite giornaliere (tutti i prodotti):")
print(vendite_giornaliere.head(), "\n")

print(f"Vendite medie giornaliere: {vendite_medie_giornaliere:.2f}")
