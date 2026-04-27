# ============================================
# Progetto 4 - Regressione su dataset Diabetes
# Pipeline completa con confronto modelli, CV,
# tuning iperparametri, learning curves e PCA
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------
# 1. Data Exploration & Preprocessing
# --------------------------------------------

# Caricamento dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="target")

df = pd.concat([X, y], axis=1)

print("Prime righe del dataset:")
print(df.head(), "\n")

print("Analisi descrittiva iniziale (df.describe()):")
print(df.describe(), "\n")

# Istogrammi delle feature
df.hist(bins=20, figsize=(12, 8))
plt.suptitle("Istogrammi delle feature", y=1.02)
plt.tight_layout()
plt.show()

# Boxplot delle feature
plt.figure(figsize=(10, 6))
df.boxplot(column=diabetes.feature_names)
plt.title("Boxplot delle feature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter matrix (solo prime 4 feature per non avere troppe combinazioni)
from pandas.plotting import scatter_matrix

plt.figure(figsize=(10, 10))
scatter_matrix(df[diabetes.feature_names[:4]], alpha=0.5, figsize=(10, 10), diagonal="hist")
plt.suptitle("Scatter matrix (prime 4 feature)", y=1.02)
plt.show()

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizzazione feature (fit SOLO sul train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# 2. Confronto e Selezione Iniziale dei Modelli
#    con K-Fold Cross-Validation e NMSE
# --------------------------------------------

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(kernel="rbf")
}

# Per i modelli che necessitano di scaling (quasi tutti), usiamo X_train_scaled
# DecisionTree è insensibile allo scaling, ma per semplicità usiamo comunque i dati scalati.

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

for name, model in models.items():
    # scoring='neg_mean_squared_error' restituisce NMSE (Negative MSE)
    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=kfold,
        scoring="neg_mean_squared_error"
    )
    cv_results.append({
        "Model": name,
        "NMSE_mean": scores.mean(),
        "NMSE_std": scores.std()
    })

cv_df = pd.DataFrame(cv_results)
print("Risultati K-Fold CV (scoring = neg_mean_squared_error):")
print(cv_df.sort_values("NMSE_mean", ascending=False), "\n")

# Selezioniamo il modello migliore in base alla NMSE media (più alta = MSE più bassa)
best_model_name = cv_df.sort_values("NMSE_mean", ascending=False).iloc[0]["Model"]
print(f"Modello selezionato per il tuning: {best_model_name}\n")

# --------------------------------------------
# 3. Hyperparameters Tuning (Grid Search)
#    + Valutazione finale su Test (MSE, R2)
# --------------------------------------------

# Scegliamo una griglia sensata per il modello migliore
if best_model_name == "Ridge":
    base_model = Ridge()
    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

elif best_model_name == "Lasso":
    base_model = Lasso(max_iter=10000)
    param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]}

elif best_model_name == "KNN":
    base_model = KNeighborsRegressor()
    param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}

elif best_model_name == "DecisionTree":
    base_model = DecisionTreeRegressor(random_state=42)
    param_grid = {"max_depth": [None, 3, 5, 7, 9]}

elif best_model_name == "SVR":
    base_model = SVR()
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1.0]
    }

else:  # LinearRegression (pochi iperparametri utili, usiamo fit_intercept)
    base_model = LinearRegression()
    param_grid = {"fit_intercept": [True, False]}

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=kfold,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("Migliori iperparametri trovati dalla Grid Search:")
print(grid_search.best_params_, "\n")

best_model = grid_search.best_estimator_

# Valutazione finale sul Test set
y_pred_test = best_model.predict(X_test_scaled)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Modello ottimizzato: {best_model}")
print(f"MSE sul Test set: {mse_test:.4f}")
print(f"R2 sul Test set:  {r2_test:.4f}\n")

# --------------------------------------------
# 4. Analisi delle Performance
#    Learning Curves per il modello ottimizzato
# --------------------------------------------

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X_train_scaled,
    y_train,
    cv=kfold,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    random_state=None
)

# Convertiamo in MSE (valori positivi)
train_mse = -train_scores
val_mse = -val_scores

train_mean = train_mse.mean(axis=1)
train_std = train_mse.std(axis=1)
val_mean = val_mse.mean(axis=1)
val_std = val_mse.std(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", color="tab:blue", label="Training MSE")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="tab:blue")
plt.plot(train_sizes, val_mean, "o-", color="tab:orange", label="Validation MSE")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="tab:orange")
plt.xlabel("Numero di campioni di training")
plt.ylabel("MSE (media su CV)")
plt.title(f"Learning curves - {best_model_name}")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# --------------------------------------------
# 5. Visualizzazione con Data Reduction (PCA)
#    PCA a 2D + plot regression
# --------------------------------------------

# PCA sulle feature scalate (train+test insieme per visualizzazione coerente)
X_all_scaled = np.vstack([X_train_scaled, X_test_scaled])
pca_2d = PCA(n_components=2)
X_all_pca = pca_2d.fit_transform(X_all_scaled)

# Ricostruiamo train/test nel nuovo spazio PCA
X_train_pca = X_all_pca[:len(X_train_scaled), :]
X_test_pca = X_all_pca[len(X_train_scaled):, :]

# Riaddestriamo il modello migliore nello spazio PCA 2D
best_model_pca = grid_search.best_estimator_
best_model_pca.fit(X_train_pca, y_train)

# Griglia 2D per visualizzare il piano di regressione
x_min, x_max = X_all_pca[:, 0].min() - 1.0, X_all_pca[:, 0].max() + 1.0
y_min, y_max = X_all_pca[:, 1].min() - 1.0, X_all_pca[:, 1].max() + 1.0

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 50),
    np.linspace(y_min, y_max, 50)
)
grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

# Predizioni del modello sul piano PCA
zz = best_model_pca.predict(grid_points_2d)
zz = zz.reshape(xx.shape)

# Scatter dei punti (train+test) con colore = target reale
plt.figure(figsize=(8, 6))
sc = plt.scatter(X_all_pca[:, 0], X_all_pca[:, 1], c=np.hstack([y_train, y_test]), cmap="viridis", alpha=0.7, edgecolor="k")
plt.colorbar(sc, label="Valore target")
plt.contourf(xx, yy, zz, alpha=0.3, cmap="viridis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Piano di regressione nel piano PCA (modello: {best_model_name})")
plt.tight_layout()
plt.show()