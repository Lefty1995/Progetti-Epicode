import os
import json
import time
from collections import Counter

import pandas as pd
import dask.dataframe as dd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ============================================================================
# CONFIGURAZIONE GENERALE
# ============================================================================
BASE_DIR = "./data_local"
PARQUET_DIR = os.path.join(BASE_DIR, "parquet")
JSON_DIR = os.path.join(BASE_DIR, "json")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_sales")


# ============================================================================
# ESERCIZIO 1 - Ingestion e Limiti di Memoria (Pandas vs Dask)
# ============================================================================

def esercizio1_pandas():
    """
    Legge tutti i file JSONL in ./data_local/json con Pandas
    e calcola la somma totale della colonna 'amount'.
    """
    total_amount = 0.0

    for filename in os.listdir(JSON_DIR):
        if not filename.endswith(".jsonl"):
            continue
        file_path = os.path.join(JSON_DIR, filename)
        print(f"[Pandas] Leggo: {file_path}")

        df = pd.read_json(file_path, lines=True)
        total_amount += df["amount"].sum()

    print(f"[Pandas] Totale generale amount: {total_amount:.2f}")


def esercizio1_dask():
    """
    Legge tutti i file JSONL in ./data_local/json con Dask
    e calcola la media di 'amount' per 'year'.
    (Se aggiungi payment_type nel generator, puoi cambiare il groupby.)
    """
    pattern = os.path.join(JSON_DIR, "*.jsonl")
    print(f"[Dask] Leggo pattern: {pattern}")

    ddf = dd.read_json(pattern, lines=True)
    grouped = ddf.groupby("year")["amount"].mean()
    result = grouped.compute()

    print("[Dask] Media amount per year:")
    print(result)


# ============================================================================
# ESERCIZIO 2 - Pipeline ETL con PySpark
# ============================================================================

def get_spark(app_name="MegaShop_App"):
    """
    Inizializza una SparkSession (riutilizzabile).
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )
    return spark


def esercizio2_etl(spark=None):
    """
    Extract-Transform-Load:
    - Legge transactions_batch_*.parquet, products.parquet, regions.parquet
    - Join per ottenere (transaction_id, region_name, category, amount, year)
    - Salva in ./data_local/processed_sales partizionato per year
    """
    close_spark = False
    if spark is None:
        spark = get_spark("MegaShop_ETL")
        close_spark = True

    transactions_path = os.path.join(PARQUET_DIR, "transactions_batch_*.parquet")
    products_path = os.path.join(PARQUET_DIR, "products.parquet")
    regions_path = os.path.join(PARQUET_DIR, "regions.parquet")

    print(f"[Spark ETL] Leggo transazioni da: {transactions_path}")
    df_trans = spark.read.parquet(transactions_path)

    print(f"[Spark ETL] Leggo prodotti da: {products_path}")
    df_prod = spark.read.parquet(products_path)

    print(f"[Spark ETL] Leggo regioni da: {regions_path}")
    df_reg = spark.read.parquet(regions_path)

    # Join con prodotti (per category) e regioni (per region_name)
    df_join1 = df_trans.join(df_prod, on="product_id", how="left")
    df_join2 = df_join1.join(df_reg, on="region_id", how="left")

    df_final = df_join2.select(
        "transaction_id",
        "region_name",
        "category",
        "amount",
        "year"
    )

    print(f"[Spark ETL] Scrivo risultato in: {PROCESSED_DIR}")
    (
        df_final
        .write
        .mode("overwrite")
        .partitionBy("year")
        .parquet(PROCESSED_DIR)
    )

    if close_spark:
        spark.stop()


# ============================================================================
# ESERCIZIO 3 - Data Visualization (Reporting)
# ============================================================================

def esercizio3_report(spark=None):
    """
    Dal DataFrame Spark pulito (processed_sales):
    - Calcola fatturato totale per categoria
    - Porta il risultato in Pandas
    - Crea un grafico a barre e lo salva come fatturato_per_categoria.png
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    close_spark = False
    if spark is None:
        spark = get_spark("MegaShop_Reporting")
        close_spark = True

    print(f"[Reporting] Leggo dati da: {PROCESSED_DIR}")
    df_final = spark.read.parquet(PROCESSED_DIR)

    df_cat = (
        df_final
        .groupBy("category")
        .agg(F.sum("amount").alias("total_revenue"))
        .orderBy("total_revenue", ascending=False)
    )

    pdf_cat = df_cat.toPandas()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=pdf_cat, x="category", y="total_revenue")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Categoria")
    plt.ylabel("Fatturato totale")
    plt.title("Fatturato per categoria")

    plt.tight_layout()
    out_path = "fatturato_per_categoria.png"
    plt.savefig(out_path, dpi=150)
    print(f"[Reporting] Grafico salvato come: {out_path}")

    if close_spark:
        spark.stop()


# ============================================================================
# ESERCIZIO 4 (Bonus) - Real-Time Streaming
# ============================================================================

class NewFileHandler(FileSystemEventHandler):
    """
    Handler per monitorare la cartella JSON_DIR.
    Ogni nuovo file .jsonl viene letto e si aggiorna il conteggio
    delle transazioni per region_id.
    """
    def __init__(self):
        super().__init__()
        self.region_counts = Counter()

    def on_created(self, event):
        if event.is_directory:
            return
        if not str(event.src_path).endswith(".jsonl"):
            return

        print(f"[Streaming] Nuovo file rilevato: {event.src_path}")
        self.process_file(event.src_path)

        print("[Streaming] Totale transazioni per region_id (aggiornato):")
        for region_id, count in self.region_counts.items():
            print(f"  region_id {region_id}: {count}")

    def process_file(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                region_id = record.get("region_id")
                if region_id is None:
                    continue
                self.region_counts[region_id] += 1


def esercizio4_streaming():
    """
    Monitoraggio in tempo reale della cartella ./data_local/json/
    usando watchdog. Ctrl+C per interrompere.
    """
    os.makedirs(JSON_DIR, exist_ok=True)

    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, JSON_DIR, recursive=False)
    observer.start()
    print(f"[Streaming] Monitoraggio in corso sulla cartella: {JSON_DIR}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

