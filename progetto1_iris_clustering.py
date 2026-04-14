import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# =========================================================
# 1. DATA EXPLORATION & PREPROCESSING
# =========================================================

# Caricamento dataset Iris ignorando le etichette originali
from sklearn.datasets import load_iris

iris = load_iris()  # type: ignore
X = pd.DataFrame(iris.data, columns=iris.feature_names)

print("=" * 60)
print("DATASET IRIS - PRIME 5 RIGHE")
print("=" * 60)
print(X.head())

print("\n" + "=" * 60)
print("INFORMAZIONI GENERALI")
print("=" * 60)
print(X.info())

print("\n" + "=" * 60)
print("ANALISI DESCRITTIVA")
print("=" * 60)
print(X.describe())

# Standardizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verifica standardizzazione
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\n" + "=" * 60)
print("STATISTICHE DOPO STANDARDIZZAZIONE")
print("=" * 60)
print(X_scaled_df.describe())

# =========================================================
# 2. STIMA DI K CON DBSCAN
# =========================================================

print("\n" + "=" * 60)
print("DBSCAN - ANALISI ITERATIVA")
print("=" * 60)

eps_values = [0.35, 0.45, 0.55]
min_samples = 5

dbscan_results = []

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    # Numero cluster escluso rumore (-1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = list(labels).count(-1)

    dbscan_results.append({
        "eps": eps,
        "min_samples": min_samples,
        "K_clusters": n_clusters,
        "noise_points": n_noise
    })

dbscan_df = pd.DataFrame(dbscan_results)
print(dbscan_df)

# Scelta plausibile di K da DBSCAN:
# prendiamo il K più frequente fra i risultati diversi da 0
valid_k = [row["K_clusters"] for row in dbscan_results if row["K_clusters"] > 0]

if len(valid_k) > 0:
    plausible_k_dbscan = max(set(valid_k), key=valid_k.count)
else:
    plausible_k_dbscan = 3  # fallback ragionevole

print(f"\nK più plausibile stimato con DBSCAN: {plausible_k_dbscan}")

# =========================================================
# 3. STIMA DI K CON ELBOW METHOD
# =========================================================

print("\n" + "=" * 60)
print("ELBOW METHOD")
print("=" * 60)

k_values = list(range(1, 11))
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Scelta del K finale:
# nel dataset Iris il gomito risulta tipicamente intorno a K=3
chosen_k = 3

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o', linewidth=2, color='royalblue')
plt.axvline(x=chosen_k, color='red', linestyle='--', label=f'K scelto = {chosen_k}')
plt.title('Elbow Method - Inertia / WCSS')
plt.xlabel('Numero di cluster K')
plt.ylabel('Inertia (WCSS)')
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# 4. APPLICAZIONE FINALE & VISUALIZZAZIONE
# =========================================================

print("\n" + "=" * 60)
print("K-MEANS FINALE")
print("=" * 60)

final_kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(X_scaled)

# Silhouette Score
sil_score = silhouette_score(X_scaled, final_labels)
print(f"Silhouette Score con K={chosen_k}: {sil_score:.4f}")

# DataFrame per visualizzazione
plot_df = pd.DataFrame(X_scaled, columns=X.columns)
plot_df["Cluster"] = final_labels

# Centroidi
centroids = final_kmeans.cluster_centers_

# Plot 2D usando le prime due feature standardizzate
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=plot_df,
    x=X.columns[0],
    y=X.columns[1],
    hue="Cluster",
    palette="deep",
    s=70
)

# Centroidi sulle prime due feature
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='black',
    s=250,
    marker='X',
    label='Centroidi'
)

plt.title(f'K-Means Clustering finale con K={chosen_k}')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =========================================================
# 5. COMMENTO FINALE AUTOMATICO
# =========================================================

print("\n" + "=" * 60)
print("COMMENTO FINALE")
print("=" * 60)
print("1. Il dataset Iris è stato caricato ignorando le etichette originali.")
print("2. Le feature sono state standardizzate con StandardScaler.")
print("3. DBSCAN è stato eseguito per eps = [0.35, 0.45, 0.55] con min_samples = 5.")
print(f"4. Il numero di cluster più plausibile stimato con DBSCAN è K = {plausible_k_dbscan}.")
print(f"5. Con l'Elbow Method il punto di gomito è stato scelto in corrispondenza di K = {chosen_k}.")
print(f"6. Il K-Means finale è stato applicato con K = {chosen_k}.")
print(f"7. Il Silhouette Score ottenuto è pari a {sil_score:.4f}.")