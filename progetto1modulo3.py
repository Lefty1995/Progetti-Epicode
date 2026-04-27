# ============================================
# Progetto Clustering - Iris
# DBSCAN + Elbow Method + KMeans
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# --------------------------------------------
# 1. Data Exploration & Preprocessing
# --------------------------------------------

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Prime righe del dataset Iris:")
print(X.head(), "\n")

print("Analisi descrittiva (df.describe()):")
print(X.describe(), "\n")

# Standardizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------
# 2. Stima di K con DBSCAN
#    min_samples = 5, eps in [0.35, 0.45, 0.55]
# --------------------------------------------

eps_values = [0.35, 0.45, 0.55]
min_samples = 5

dbscan_results = []

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    # Cluster = etichette diverse da -1
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    n_noise = np.sum(labels == -1)

    dbscan_results.append({
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise
    })

dbscan_df = pd.DataFrame(dbscan_results)
print("Risultati DBSCAN (stima di K):")
print(dbscan_df, "\n")

# Scelta di K "più plausibile" dai risultati DBSCAN
# (Qui puoi scegliere a mano in base a n_clusters e n_noise;
#  ad esempio prendi il valore con un numero di cluster ragionevole
#  e pochi punti rumorosi)
# Per semplicità, supponiamo di scegliere il K massimo trovato > 0
plausible = dbscan_df[dbscan_df["n_clusters"] > 0]
if not plausible.empty:
    K_dbscan = int(plausible.sort_values("n_clusters", ascending=False).iloc[0]["n_clusters"])
else:
    K_dbscan = 3  # fallback

print(f"K scelto da DBSCAN (euristico): {K_dbscan}\n")

# --------------------------------------------
# 3. Stima di K con Elbow Method (KMeans, K=1..10)
# --------------------------------------------

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, "o-", color="tab:blue")
plt.xlabel("K (numero di cluster)")
plt.ylabel("Inerzia (WCSS)")
plt.title("Elbow Method - KMeans su Iris")
plt.grid(alpha=0.3)

# Qui decidi il K dal grafico (visivamente).
# Per Iris è ragionevole K=3, ma puoi cambiare.
K_elbow = 3
plt.axvline(x=K_elbow, color="tab:red", linestyle="--", label=f"K scelto = {K_elbow}")
plt.legend()
plt.tight_layout()
plt.show()

print(f"K scelto dall'Elbow Method (a vista): {K_elbow}\n")

# --------------------------------------------
# 4. Applicazione Finale & Visualizzazione
#    - Scelta K finale
#    - KMeans, Silhouette
#    - Plot 2D con centroidi (prime due feature)
# --------------------------------------------

# Decidi come combinare K_dbscan e K_elbow.
# Opzione semplice: usare K_elbow come K finale.
K_final = K_elbow
print(f"K finale utilizzato per KMeans: {K_final}\n")

kmeans_final = KMeans(n_clusters=K_final, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Silhouette Score (solo se K_final > 1)
if K_final > 1:
    sil_score = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette Score per KMeans con K={K_final}: {sil_score:.4f}\n")
else:
    print("Silhouette Score non definito per K=1.\n")

# Visualizzazione 2D: prime due feature
X_2d = X_scaled[:, :2]          # sepal length, sepal width
centroids_2d = kmeans_final.cluster_centers_[:, :2]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=cluster_labels,
    cmap="viridis",
    alpha=0.7,
    edgecolor="k"
)
plt.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    s=200,
    c="red",
    marker="X",
    label="Centroidi"
)
plt.xlabel("Feature 1 (sepal length, standardizzata)")
plt.ylabel("Feature 2 (sepal width, standardizzata)")
plt.title(f"KMeans su Iris (K={K_final}) - prime due feature")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()