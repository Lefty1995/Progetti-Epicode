# ============================================
# Progetto 2 - Machine Learning
# Dataset Linnerud: regressione Multi-Target
# ridotta a Single-Target con PCA e selezione manuale
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_linnerud
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

from IPython.display import display

# --------------------------------------------
# 1. Data Exploration & Preprocessing
# --------------------------------------------

# Caricamento dataset
lin = load_linnerud()

X = pd.DataFrame(lin.data, columns=lin.feature_names)      
Y = pd.DataFrame(lin.target, columns=lin.target_names)     
df = pd.concat([X, Y], axis=1)

print("Feature (X):")
display(X.head())

print("Target (Y):")
display(Y.head())

print("Analisi descrittiva iniziale (X+Y):")
display(df.describe())

# Standardizzazione delle feature X
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Standardizzazione dei target Y
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y)
Y_scaled_df = pd.DataFrame(Y_scaled, columns=Y.columns)

print("Feature standardizzate (X_scaled):")
display(X_scaled_df.head())

print("Target standardizzati (Y_scaled):")
display(Y_scaled_df.head())

# --------------------------------------------
# 2. Target Engineering
#    - PCA sui target (PC1)
#    - Selezione manuale target 'Waist'
# --------------------------------------------

# PCA sui target standardizzati, mantengo solo la prima componente
pca_y = PCA(n_components=1)
y_pca = pca_y.fit_transform(Y_scaled).ravel()   # Target composto (PC1)

print("Varianza spiegata da PC1 dei target:", pca_y.explained_variance_ratio_[0])

# Target manuale: 'Waist' standardizzato
y_waist = Y_scaled_df['Waist']

# Giusto per vedere i due vettori target
display(pd.DataFrame({
    'Target_PCA_PC1': y_pca,
    'Target_Waist_std': y_waist
}))

# --------------------------------------------
# 3. Addestramento & Valutazione
#    Modelli: Linear Regression, Ridge, Lasso
#    Scenari: PCA target vs Waist
#    Training e valutazione su tutto il dataset
# --------------------------------------------

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1, max_iter=10000)
}

results = []

for scenario_name, y in [('PCA target (PC1)', y_pca),
                        ('Manual target (Waist)', y_waist)]:
    for model_name, model in models.items():
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        results.append({
            'Scenario': scenario_name,
            'Model': model_name,
            'MSE': mse,
            'R2': r2
        })

results_df = pd.DataFrame(results)
print("Risultati (MSE e R2) per modello e scenario:")
display(results_df)

# --------------------------------------------
# 4. Analisi & Visualizzazione
#    - PCA sulle feature (PC1, PC2)
#    - Regressione su sola PC1 (feature)
#      verso target PCA (PC1 dei target)
#    - Rette: Linear, Ridge, Lasso con R2 in etichetta
# --------------------------------------------

# PCA sulle feature standardizzate, tengo 2 componenti per visualizzazione
pca_X = PCA(n_components=2)
X_pca_2d = pca_X.fit_transform(X_scaled)
pc1 = X_pca_2d[:, 0]
pc2 = X_pca_2d[:, 1]

# Matrice per regressione solo su PC1 (feature)
X_pc1 = pc1.reshape(-1, 1)

plot_models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1, max_iter=10000)
}

plt.figure(figsize=(10, 6))

# Scatter: asse X = PC1 delle feature, asse Y = target PCA (PC1 dei target)
plt.scatter(pc1, y_pca, color='black', s=55, alpha=0.8,
            label='Dati reali (target PCA)')

# Per disegnare le rette ordinate lungo PC1
order = np.argsort(pc1)
colors = ['tab:blue', 'tab:orange', 'tab:green']

for (model_name, model), color in zip(plot_models.items(), colors):
    model.fit(X_pc1, y_pca)
    pred = model.predict(X_pc1)
    r2 = r2_score(y_pca, pred)

    plt.plot(pc1[order], pred[order], color=color, linewidth=2.2,
            label=f'{model_name} (R²={r2:.3f})')

plt.xlabel('PC1 delle feature standardizzate')
plt.ylabel('Target composto: PCA(target) con n_components=1')
plt.title('Regressione su PC1 delle feature vs target PCA')
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()