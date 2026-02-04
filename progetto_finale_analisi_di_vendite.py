# ============================================
# Progetto Finale – Analisi di Vendite in una Catena di Negozi
# Un unico file Python che:
# - genera un CSV di partenza (vendite.csv) con dati fittizi
# - importa i dati con Pandas
# - usa NumPy per analisi numeriche
# - usa Matplotlib per grafici
# - salva un CSV finale (vendite_analizzate.csv)
# ============================================

import os
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# PARTE 1 – CREAZIONE DATASET DI BASE (vendite.csv)
# =====================================================

def genera_vendite_csv(nome_file="vendite.csv", num_righe=30):
    """
    Crea un file CSV chiamato vendite.csv con almeno 30 righe e colonne:
    Data (YYYY-MM-DD), Negozio, Prodotto, Quantità, Prezzo_unitario.
    I dati sono inventati, ma coerenti con lo scenario.
    """
    # Negozi della catena
    negozi = ["Milano", "Roma", "Napoli", "Torino", "Bologna"]

    # Prodotti venduti
    prodotti = ["Smartphone", "Laptop", "TV", "Tablet", "Cuffie", "Console"]

    # Data di partenza per simulare vendite giornaliere
    data_inizio = datetime(2023, 9, 1)

    righe = []

    for i in range(num_righe):
        # data casuale nei 30 giorni successivi
        data = data_inizio + timedelta(days=random.randint(0, 29))
        negozio = random.choice(negozi)
        prodotto = random.choice(prodotti)

        # quantità venduta tra 1 e 15 pezzi
        quantita = random.randint(1, 15)

        # prezzo unitario in base al tipo di prodotto
        if prodotto == "Smartphone":
            prezzo_unitario = round(random.uniform(299.99, 999.99), 2)
        elif prodotto == "Laptop":
            prezzo_unitario = round(random.uniform(499.99, 1499.99), 2)
        elif prodotto == "TV":
            prezzo_unitario = round(random.uniform(399.99, 1999.99), 2)
        elif prodotto == "Tablet":
            prezzo_unitario = round(random.uniform(199.99, 799.99), 2)
        elif prodotto == "Cuffie":
            prezzo_unitario = round(random.uniform(29.99, 299.99), 2)
        else:  # Console
            prezzo_unitario = round(random.uniform(249.99, 699.99), 2)

        righe.append(
            [
                data.strftime("%Y-%m-%d"),
                negozio,
                prodotto,
                quantita,
                prezzo_unitario,
            ]
        )

    # Creazione DataFrame e salvataggio CSV
    df = pd.DataFrame(
        righe,
        columns=["Data", "Negozio", "Prodotto", "Quantità", "Prezzo_unitario"],
    )
    df.to_csv(nome_file, index=False)
    print(f"Creato il file CSV di base: {nome_file}")


# =====================================================
# PARTE 2 – IMPORTAZIONE CSV CON PANDAS
# =====================================================

def importa_vendite_csv(nome_file="vendite.csv"):
    """
    Importa il file CSV in un DataFrame Pandas e stampa:
    - le prime 5 righe
    - shape (righe, colonne)
    - info()
    Restituisce il DataFrame per le parti successive.
    """
    df = pd.read_csv(nome_file)

    print("\n=== Parte 2 – Importazione CSV con Pandas ===")
    print("Prime 5 righe del DataFrame:")
    print(df.head())

    print("\nShape (righe, colonne):")
    print(df.shape)

    print("\nInformazioni generali del DataFrame:")
    print(df.info())

    return df


# =====================================================
# PARTE 3 – ELABORAZIONI CON PANDAS
# =====================================================

def elaborazioni_pandas(df):
    """
    Aggiunge la colonna 'Incasso' e calcola:
    - incasso totale catena
    - incasso medio per negozio
    - 3 prodotti più venduti per quantità totale
    - incasso medio raggruppato per (Negozio, Prodotto)
    """
    print("\n=== Parte 3 – Elaborazioni con Pandas ===")

    # Colonna Incasso = Quantità * Prezzo_unitario
    df["Incasso"] = df["Quantità"] * df["Prezzo_unitario"]

    # Incasso totale catena
    incasso_totale = df["Incasso"].sum()
    print(f"\nIncasso totale catena: {incasso_totale:.2f} €")

    # Incasso medio per negozio
    incasso_medio_negozio = df.groupby("Negozio")["Incasso"].mean()
    print("\nIncasso medio per negozio:")
    print(incasso_medio_negozio)

    # 3 prodotti più venduti (per quantità totale)
    quantita_per_prodotto = df.groupby("Prodotto")["Quantità"].sum()
    top3_prodotti = quantita_per_prodotto.sort_values(ascending=False).head(3)
    print("\nTop 3 prodotti più venduti (per quantità totale):")
    print(top3_prodotti)

    # Incasso medio raggruppato per Negozio e Prodotto
    incasso_medio_negozio_prodotto = (
        df.groupby(["Negozio", "Prodotto"])["Incasso"].mean()
    )
    print("\nIncasso medio per (Negozio, Prodotto):")
    print(incasso_medio_negozio_prodotto)

    return df


# =====================================================
# PARTE 4 – USO DI NUMPY
# =====================================================

def analisi_numpy(df):
    """
    Usa NumPy sulla colonna Quantità:
    - media, minimo, massimo, deviazione standard
    - percentuale di vendite sopra la media

    Crea anche un array 2D con (Quantità, Prezzo_unitario)
    e calcola l'incasso per riga, confrontandolo con df['Incasso'].
    """
    print("\n=== Parte 4 – Uso di NumPy ===")

    # Estrarre Quantità come array NumPy
    q = df["Quantità"].to_numpy()

    media = np.mean(q)
    minimo = np.min(q)
    massimo = np.max(q)
    dev_std = np.std(q)

    print(f"Quantità - media: {media:.2f}, min: {minimo}, max: {massimo}, dev.std: {dev_std:.2f}")

    # Percentuale di vendite sopra la media
    num_sopra_media = np.sum(q > media)
    percentuale_sopra_media = (num_sopra_media / len(q)) * 100
    print(f"Percentuale di righe con Quantità sopra la media: {percentuale_sopra_media:.2f}%")

    # Array 2D: colonne [Quantità, Prezzo_unitario]
    quantita_array = df["Quantità"].to_numpy()
    prezzo_array = df["Prezzo_unitario"].to_numpy()
    matrice_q_prezzo = np.column_stack((quantita_array, prezzo_array))

    # Incasso calcolato con NumPy riga per riga (Quantità * Prezzo_unitario)
    incasso_numpy = matrice_q_prezzo[:, 0] * matrice_q_prezzo[:, 1]

    # Confronto con la colonna Incasso del DataFrame
    incasso_dataframe = df["Incasso"].to_numpy()

    # Verifica se tutti i valori coincidono (entro una piccola tolleranza)
    coincidenza = np.allclose(incasso_numpy, incasso_dataframe)
    print(f"I valori di incasso calcolati con NumPy coincidono con la colonna Incasso? {coincidenza}")


# =====================================================
# PARTE 5 – VISUALIZZAZIONI CON MATPLOTLIB
# =====================================================

def grafici_base(df):
    """
    Crea:
    - grafico a barre: incasso totale per ogni negozio
    - grafico a torta: percentuale incassi per prodotto
    - grafico a linee: andamento giornaliero degli incassi totali
    """
    print("\n=== Parte 5 – Visualizzazioni con Matplotlib ===")

    # Assicuriamoci che la colonna Data sia di tipo datetime
    if df["Data"].dtype == "object":
        df["Data"] = pd.to_datetime(df["Data"])

    # Grafico a barre: incasso totale per negozio
    incasso_per_negozio = df.groupby("Negozio")["Incasso"].sum()
    plt.figure(figsize=(8, 5))
    incasso_per_negozio.plot(kind="bar", color="skyblue")
    plt.title("Incasso totale per negozio")
    plt.xlabel("Negozio")
    plt.ylabel("Incasso (€)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Grafico a torta: percentuale incassi per prodotto
    incasso_per_prodotto = df.groupby("Prodotto")["Incasso"].sum()
    plt.figure(figsize=(6, 6))
    plt.pie(
        incasso_per_prodotto.values,
        labels=incasso_per_prodotto.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Percentuale incassi per prodotto")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Grafico a linee: andamento giornaliero incassi totali
    incasso_giornaliero = df.groupby("Data")["Incasso"].sum().sort_index()
    plt.figure(figsize=(8, 5))
    plt.plot(incasso_giornaliero.index, incasso_giornaliero.values, marker="o")
    plt.title("Andamento giornaliero incassi totali")
    plt.xlabel("Data")
    plt.ylabel("Incasso giornaliero (€)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =====================================================
# PARTE 6 – ANALISI AVANZATA CON CATEGORIE
# =====================================================

def aggiungi_categoria(df):
    """
    Crea una nuova colonna 'Categoria' raggruppando i prodotti
    in famiglie (esempio):
    - Smartphone, Laptop, Tablet -> Informatica
    - TV -> Elettrodomestici
    - Cuffie, Console -> Accessori / Gaming
    """
    mappa_categoria = {
        "Smartphone": "Informatica",
        "Laptop": "Informatica",
        "Tablet": "Informatica",
        "TV": "Elettrodomestici",
        "Cuffie": "Accessori",
        "Console": "Gaming",
    }

    df["Categoria"] = df["Prodotto"].map(mappa_categoria).fillna("Altro")
    return df


def analisi_per_categoria(df):
    """
    Per ogni categoria calcola:
    - incasso totale
    - quantità media venduta
    """
    print("\n=== Parte 6 – Analisi avanzata per categoria ===")

    incasso_per_categoria = df.groupby("Categoria")["Incasso"].sum()
    print("\nIncasso totale per categoria:")
    print(incasso_per_categoria)

    quantita_media_categoria = df.groupby("Categoria")["Quantità"].mean()
    print("\nQuantità media venduta per categoria:")
    print(quantita_media_categoria)


def salva_vendite_analizzate(df, nome_file="vendite_analizzate.csv"):
    """
    Salva il DataFrame aggiornato (con Incasso e Categoria)
    in un nuovo file CSV vendite_analizzate.csv.
    """
    df.to_csv(nome_file, index=False)
    print(f"\nDataFrame analizzato salvato in: {nome_file}")


# =====================================================
# PARTE 7 – ESTENSIONI
# =====================================================

def grafico_combinato_categoria(df):
    """
    Grafico combinato:
    - barre: incasso medio per categoria
    - linea: quantità media venduta per categoria
    """
    print("\n=== Parte 7 – Grafico combinato per categoria ===")

    incasso_medio_categoria = df.groupby("Categoria")["Incasso"].mean()
    quantita_media_categoria = df.groupby("Categoria")["Quantità"].mean()

    categorie = incasso_medio_categoria.index
    valori_incasso = incasso_medio_categoria.values
    valori_quantita = quantita_media_categoria.values

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Barre: incasso medio per categoria
    ax1.bar(categorie, valori_incasso, color="lightgreen", label="Incasso medio (€)")
    ax1.set_xlabel("Categoria")
    ax1.set_ylabel("Incasso medio (€)", color="green")
    ax1.tick_params(axis="y", labelcolor="green")

    # Linea: quantità media per categoria
    ax2 = ax1.twinx()
    ax2.plot(categorie, valori_quantita, color="blue", marker="o", label="Quantità media")
    ax2.set_ylabel("Quantità media", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title("Incasso medio e quantità media per categoria")
    fig.tight_layout()
    plt.show()


def top_n_prodotti(df, n=3):
    """
    Restituisce i n prodotti più venduti in termini di incasso totale.
    """
    incasso_per_prodotto = df.groupby("Prodotto")["Incasso"].sum()
    top = incasso_per_prodotto.sort_values(ascending=False).head(n)
    return top


# =====================================================
# BLOCCO PRINCIPALE – ESECUZIONE COMPLETA
# =====================================================

if __name__ == "__main__":
    # Se il CSV non esiste, lo generiamo (Parte 1)
    if not os.path.exists("vendite.csv"):
        genera_vendite_csv("vendite.csv", num_righe=30)

    # Parte 2: importazione
    df_vendite = importa_vendite_csv("vendite.csv")

    # Parte 3: elaborazioni con Pandas (aggiunge Incasso)
    df_vendite = elaborazioni_pandas(df_vendite)

    # Parte 4: analisi NumPy
    analisi_numpy(df_vendite)

    # Parte 5: grafici base con Matplotlib
    grafici_base(df_vendite)

    # Parte 6: categorie + analisi + salvataggio CSV finale
    df_vendite = aggiungi_categoria(df_vendite)
    analisi_per_categoria(df_vendite)
    salva_vendite_analizzate(df_vendite, "vendite_analizzate.csv")

    # Parte 7: estensioni
    grafico_combinato_categoria(df_vendite)

    print("\nTop 3 prodotti per incasso totale:")
    print(top_n_prodotti(df_vendite, n=3))
