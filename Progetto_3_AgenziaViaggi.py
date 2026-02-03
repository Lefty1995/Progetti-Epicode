# ============================================
# Progetto d'Esame – Sistema Prenotazione Viaggi
# Unico file Python con:
# - Parte 1: variabili base
# - Parte 2: classi OOP (Cliente, Viaggio, Prenotazione)
# - Parte 3: analisi NumPy
# - Parte 4–6: analisi Pandas + grafici Matplotlib
# - Parte 7: estensioni (top clienti + grafico combinato)
# ============================================

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# PARTE 1 – VARIABILI E TIPI DI DATI (richiesta base)
# =====================================================

# Definizione delle informazioni di un singolo cliente di esempio
nome = "Mario Rossi"
eta = 34
saldo = 2500.75
vip = True

# Lista di destinazioni disponibili (almeno 5 città)
destinazioni_disponibili = [
    "Roma",
    "Parigi",
    "New York",
    "Tokyo",
    "Cairo",
    "Londra",
    "Buenos Aires",
]

# Dizionario che associa ogni destinazione a un prezzo medio del viaggio
prezzi_medi_viaggi = {
    "Roma": 350.0,
    "Parigi": 600.0,
    "New York": 1200.0,
    "Tokyo": 1500.0,
    "Cairo": 800.0,
    "Londra": 700.0,
    "Buenos Aires": 1100.0,
}


# =====================================================
# PARTE 2 – PROGRAMMAZIONE AD OGGETTI (OOP)
# =====================================================

class Cliente:
    """
    Rappresenta un cliente dell'agenzia di viaggi.
    Attributi:
        nome (str)
        eta (int)
        vip (bool)
    """

    def __init__(self, nome: str, eta: int, vip: bool = False):
        self.nome = nome
        self.eta = eta
        self.vip = vip

    def stampa_info(self):
        """Stampa le informazioni principali del cliente."""
        stato_vip = "VIP" if self.vip else "Standard"
        print(f"Cliente: {self.nome}, Età: {self.eta}, Stato: {stato_vip}")


class Viaggio:
    """
    Rappresenta un viaggio prenotabile.
    Attributi:
        destinazione (str)
        prezzo (float)
        durata_giorni (int)
    """

    def __init__(self, destinazione: str, prezzo: float, durata_giorni: int):
        self.destinazione = destinazione
        self.prezzo = prezzo
        self.durata_giorni = durata_giorni


class Prenotazione:
    """
    Collega un Cliente a un Viaggio.
    Calcola l'importo finale con sconto del 10% se il cliente è VIP.
    Attributi:
        cliente (Cliente)
        viaggio (Viaggio)
        giorno_partenza (datetime)
        importo_finale (float)
    """

    def __init__(self, cliente: Cliente, viaggio: Viaggio, giorno_partenza: datetime):
        self.cliente = cliente
        self.viaggio = viaggio
        self.giorno_partenza = giorno_partenza
        self.importo_finale = self.calcola_importo_finale()

    def calcola_importo_finale(self) -> float:
        """Calcola l'importo con eventuale sconto VIP del 10%."""
        prezzo_base = self.viaggio.prezzo
        if self.cliente.vip:
            return prezzo_base * 0.9  # sconto 10%
        return prezzo_base

    def dettagli(self):
        """Stampa le informazioni complete della prenotazione."""
        print("=== Dettagli Prenotazione ===")
        self.cliente.stampa_info()
        print(f"Destinazione: {self.viaggio.destinazione}")
        print(f"Durata: {self.viaggio.durata_giorni} giorni")
        print(f"Giorno partenza: {self.giorno_partenza.strftime('%Y-%m-%d')}")
        print(f"Importo finale: {self.importo_finale:.2f} €")
        print("=============================")


# =====================================================
# DATI DI BASE: CREAZIONE CLIENTI E VIAGGI (5 clienti)
# =====================================================

def crea_clienti_di_base():
    """Crea 5 clienti inventati, alcuni VIP e altri no."""
    clienti = [
        Cliente("Mario Rossi", 34, vip=True),
        Cliente("Anna Bianchi", 28, vip=False),
        Cliente("Luca Verdi", 45, vip=True),
        Cliente("Giulia Neri", 31, vip=False),
        Cliente("Paolo Esposito", 52, vip=False),
    ]
    return clienti


def crea_viaggi_di_base():
    """Crea oggetti Viaggio a partire dal dizionario prezzi_medi_viaggi."""
    viaggi = []
    for dest, prezzo in prezzi_medi_viaggi.items():
        # Durata casuale tra 3 e 14 giorni
        durata = random.randint(3, 14)
        viaggi.append(Viaggio(destinazione=dest, prezzo=prezzo, durata_giorni=durata))
    return viaggi


# =====================================================
# PARTE 3 – NUMPY: 100 PRENOTAZIONI SIMULATE
# =====================================================

def analisi_numpy_prezzi():
    """
    Genera un array NumPy di 100 prenotazioni simulate
    con prezzi casuali fra 200 e 2000 €, e calcola:
    - prezzo medio
    - minimo, massimo
    - deviazione standard
    - percentuale di prenotazioni sopra la media
    """
    prezzi = np.random.uniform(200, 2000, size=100)  # 100 float casuali

    prezzo_medio = np.mean(prezzi)
    prezzo_min = np.min(prezzi)
    prezzo_max = np.max(prezzi)
    dev_std = np.std(prezzi)

    # Percentuale di prenotazioni con prezzo > media
    num_sopra_media = np.sum(prezzi > prezzo_medio)
    percentuale_sopra_media = (num_sopra_media / len(prezzi)) * 100

    print("=== Analisi NumPy su 100 prenotazioni simulate ===")
    print(f"Prezzo medio: {prezzo_medio:.2f} €")
    print(f"Prezzo minimo: {prezzo_min:.2f} €")
    print(f"Prezzo massimo: {prezzo_max:.2f} €")
    print(f"Deviazione standard: {dev_std:.2f}")
    print(f"Percentuale prenotazioni sopra la media: {percentuale_sopra_media:.2f}%")
    print("=================================================")


# =====================================================
# PARTE 4 – PANDAS: CREAZIONE DATAFRAME PRENOTAZIONI
# =====================================================

def genera_prenotazioni(clienti, viaggi, num_prenotazioni=80):
    """
    Genera una lista di oggetti Prenotazione.
    - clienti: lista di Cliente
    - viaggi: lista di Viaggio
    - num_prenotazioni: numero di prenotazioni da creare
    Le date di partenza sono distribuite sui prossimi 30 giorni.
    """
    prenotazioni = []
    oggi = datetime.today()

    for _ in range(num_prenotazioni):
        cliente = random.choice(clienti)
        viaggio = random.choice(viaggi)
        delta_giorni = random.randint(0, 29)
        giorno_partenza = oggi + timedelta(days=delta_giorni)
        pren = Prenotazione(cliente, viaggio, giorno_partenza)
        prenotazioni.append(pren)

    return prenotazioni


def crea_dataframe_prenotazioni(prenotazioni):
    """
    Crea un DataFrame Pandas con colonne:
    Cliente, Destinazione, Prezzo, Giorno_Partenza, Durata, Incasso
    """
    dati = {
        "Cliente": [],
        "Destinazione": [],
        "Prezzo": [],
        "Giorno_Partenza": [],
        "Durata": [],
        "Incasso": [],
    }

    for pren in prenotazioni:
        dati["Cliente"].append(pren.cliente.nome)
        dati["Destinazione"].append(pren.viaggio.destinazione)
        dati["Prezzo"].append(pren.viaggio.prezzo)
        dati["Giorno_Partenza"].append(pren.giorno_partenza.date())
        dati["Durata"].append(pren.viaggio.durata_giorni)
        dati["Incasso"].append(pren.importo_finale)

    df = pd.DataFrame(dati)
    return df


def analisi_pandas_base(df):
    """
    Calcola con Pandas:
    - incasso totale dell’agenzia
    - incasso medio per destinazione
    - top 3 destinazioni più vendute (per numero di prenotazioni)
    """
    print("=== Analisi Pandas – Statistiche di base ===")

    incasso_totale = df["Incasso"].sum()
    print(f"Incasso totale agenzia: {incasso_totale:.2f} €")

    incasso_medio_dest = df.groupby("Destinazione")["Incasso"].mean()
    print("\nIncasso medio per destinazione:")
    print(incasso_medio_dest)

    top3_dest = df["Destinazione"].value_counts().head(3)
    print("\nTop 3 destinazioni più vendute (per numero di prenotazioni):")
    print(top3_dest)

    print("===========================================")


# =====================================================
# PARTE 5 – MATPLOTLIB: GRAFICI
# =====================================================

def grafico_incasso_per_destinazione(df):
    """Grafico a barre: incasso totale per ogni destinazione."""
    incasso_per_dest = df.groupby("Destinazione")["Incasso"].sum()

    plt.figure(figsize=(8, 5))
    plt.bar(incasso_per_dest.index, incasso_per_dest.values, color="skyblue")
    plt.title("Incasso totale per destinazione")
    plt.xlabel("Destinazione")
    plt.ylabel("Incasso (€)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def grafico_andamento_giornaliero(df):
    """Grafico a linee: andamento giornaliero degli incassi."""
    incasso_giornaliero = df.groupby("Giorno_Partenza")["Incasso"].sum().sort_index()

    plt.figure(figsize=(8, 5))
    plt.plot(incasso_giornaliero.index, incasso_giornaliero.values, marker="o")
    plt.title("Andamento giornaliero degli incassi")
    plt.xlabel("Giorno di partenza")
    plt.ylabel("Incasso giornaliero (€)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def grafico_torta_vendite_per_destinazione(df):
    """Grafico a torta: percentuale di vendite per ciascuna destinazione (per numero di prenotazioni)."""
    vendite_per_dest = df["Destinazione"].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(
        vendite_per_dest.values,
        labels=vendite_per_dest.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Percentuale di vendite per destinazione")
    plt.axis("equal")  # rende il grafico circolare
    plt.tight_layout()
    plt.show()


# =====================================================
# PARTE 6 – ANALISI AVANZATA PER CATEGORIA
# =====================================================

def aggiungi_categoria_viaggio(df):
    """
    Aggiunge una colonna 'Categoria' al DataFrame, raggruppando
    le destinazioni in: 'Europa', 'Asia', 'America', 'Africa'.
    Usiamo un dizionario di mapping.
    """
    mappa_categoria = {
        "Roma": "Europa",
        "Parigi": "Europa",
        "Londra": "Europa",
        "Cairo": "Africa",
        "New York": "America",
        "Buenos Aires": "America",
        "Tokyo": "Asia",
    }

    df["Categoria"] = df["Destinazione"].map(mappa_categoria).fillna("Altro")
    return df


def analisi_pandas_categoria(df):
    """
    Calcola:
    - incasso totale per categoria
    - durata media dei viaggi per categoria
    """
    print("=== Analisi Pandas – Per categoria ===")

    incasso_per_categoria = df.groupby("Categoria")["Incasso"].sum()
    print("\nIncasso totale per categoria:")
    print(incasso_per_categoria)

    durata_media_categoria = df.groupby("Categoria")["Durata"].mean()
    print("\nDurata media viaggi per categoria (giorni):")
    print(durata_media_categoria)

    print("=======================================")


def salva_dataframe_csv(df, nome_file="prenotazioni_analizzate.csv"):
    """
    Salva il DataFrame aggiornato in un file CSV.
    """
    df.to_csv(nome_file, index=False)
    print(f"DataFrame salvato in: {nome_file}")


# =====================================================
# PARTE 7 – ESTENSIONI
# =====================================================

def top_n_clienti_per_prenotazioni(df, n=3):
    """
    Restituisce i N clienti con più prenotazioni.
    Ritorna una Series ordinata (nome_cliente -> numero_prenotazioni).
    """
    conteggio = df["Cliente"].value_counts().head(n)
    return conteggio


def grafico_combinato_categoria(df):
    """
    Grafico combinato (barre + linea) che mostri:
    - barre = incasso medio per categoria
    - linea = durata media per categoria
    """
    incasso_medio_categoria = df.groupby("Categoria")["Incasso"].mean()
    durata_media_categoria = df.groupby("Categoria")["Durata"].mean()

    categorie = incasso_medio_categoria.index
    valori_incasso = incasso_medio_categoria.values
    valori_durata = durata_media_categoria.values

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Barre: incasso medio per categoria (asse di sinistra)
    ax1.bar(categorie, valori_incasso, color="lightgreen", label="Incasso medio (€)")
    ax1.set_xlabel("Categoria")
    ax1.set_ylabel("Incasso medio (€)", color="green")
    ax1.tick_params(axis="y", labelcolor="green")

    # Linea: durata media per categoria (asse di destra)
    ax2 = ax1.twinx()
    ax2.plot(categorie, valori_durata, color="blue", marker="o", label="Durata media (giorni)")
    ax2.set_ylabel("Durata media (giorni)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title("Incasso medio e durata media per categoria")
    fig.tight_layout()
    plt.show()


# =====================================================
# BLOCCO PRINCIPALE – ESECUZIONE DEL PROGETTO
# =====================================================

if __name__ == "__main__":
    # --- Creazione dati di base ---
    clienti = crea_clienti_di_base()
    viaggi = crea_viaggi_di_base()

    # Esempio OOP: una prenotazione singola con dettagli()
    pren_demo = Prenotazione(clienti[0], viaggi[0], datetime.today())
    pren_demo.dettagli()

    # --- Parte 3: analisi NumPy ---
    analisi_numpy_prezzi()

    # --- Parte 4: generazione prenotazioni + DataFrame ---
    prenotazioni = genera_prenotazioni(clienti, viaggi, num_prenotazioni=80)
    df_prenotazioni = crea_dataframe_prenotazioni(prenotazioni)

    print("\nPrime righe del DataFrame prenotazioni:")
    print(df_prenotazioni.head())

    analisi_pandas_base(df_prenotazioni)

    # --- Parte 6: aggiunta categoria + analisi avanzata ---
    df_prenotazioni = aggiungi_categoria_viaggio(df_prenotazioni)
    analisi_pandas_categoria(df_prenotazioni)

    # Salvataggio CSV richiesto
    salva_dataframe_csv(df_prenotazioni, "prenotazioni_analizzate.csv")

    # --- Parte 5: grafici singoli ---
    grafico_incasso_per_destinazione(df_prenotazioni)
    grafico_andamento_giornaliero(df_prenotazioni)
    grafico_torta_vendite_per_destinazione(df_prenotazioni)

    # --- Parte 7: estensioni ---
    print("\nTop 3 clienti con più prenotazioni:")
    print(top_n_clienti_per_prenotazioni(df_prenotazioni, n=3))

    grafico_combinato_categoria(df_prenotazioni)
