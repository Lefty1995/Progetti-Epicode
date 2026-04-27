# ============================================
# Progetto 3 - Machine Learning
# Dataset Wine: PCA + KNN vs SVM
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --------------------------------------------
# 1. Data Exploration & Preprocessing
# --------------------------------------------

# Caricamento dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="target")

df = pd.concat([X, y], axis=1)

print("Prime righe del dataset:")
print(df.head(), "\n")

print("Analisi descrittiva iniziale (df.describe()):")
print(df.describe(), "\n")

print("Tipi delle colonne (df.dtypes):")
print(df.dtypes, "\n")

# Standardizzazione delle feature X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------
# 2. Data Dimensionality Reduction (PCA a 2 componenti)
# --------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Shape originale di X:", X.shape)
print("Shape dopo PCA (2 componenti):", X_pca.shape, "\n")

# --------------------------------------------
# 3. Addestramento & Valutazione
#    - KNN (n_neighbors=5)
#    - SVM (kernel='linear')
#    Training e test sull'intero set ridotto
# --------------------------------------------

# Modelli
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel="linear")

# Addestramento sui dati ridotti (2D)
knn.fit(X_pca, y)
svm.fit(X_pca, y)

# Predizioni sugli stessi dati
y_pred_knn = knn.predict(X_pca)
y_pred_svm = svm.predict(X_pca)

# Accuracy
acc_knn = accuracy_score(y, y_pred_knn)
acc_svm = accuracy_score(y, y_pred_svm)

print(f"Accuracy KNN (n_neighbors=5)  su X_pca: {acc_knn:.4f}")
print(f"Accuracy SVM (kernel='linear') su X_pca: {acc_svm:.4f}\n")

# --------------------------------------------
# 4. Analisi & Visualizzazione
#    - Due subplot: KNN vs SVM
#    - Data points colorati per classe
#    - Decision boundary dei modelli
# --------------------------------------------

# Creiamo una griglia 2D su cui valutare i modelli
x_min, x_max = X_pca[:, 0].min() - 1.0, X_pca[:, 0].max() + 1.0
y_min, y_max = X_pca[:, 1].min() - 1.0, X_pca[:, 1].max() + 1.0

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predizioni sulla griglia
Z_knn = knn.predict(grid_points).reshape(xx.shape)
Z_svm = svm.predict(grid_points).reshape(xx.shape)

# Class labels
class_names = wine.target_names

plt.figure(figsize=(12, 5))

# ---- Subplot 1: KNN ----
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_knn, alpha=0.3, cmap=plt.cm.Set1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.title(f"KNN (k=5) - Accuracy: {acc_knn:.3f}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(alpha=0.2)

# legenda classi
handles, _ = scatter.legend_elements()
plt.legend(handles, class_names, title="Classi")

# ---- Subplot 2: SVM ----
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_svm, alpha=0.3, cmap=plt.cm.Set1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.title(f"SVM (kernel='linear') - Accuracy: {acc_svm:.3f}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(alpha=0.2)

handles, _ = scatter.legend_elements()
plt.legend(handles, class_names, title="Classi")

plt.tight_layout()
plt.show()