# Définissons d'abord les fonctions nécessaires pour exécuter le script complet,
# y compris la lecture des données, le calcul des paramètres, la prédiction, la visualisation,
# et la validation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chargement des données
df_train = pd.read_csv('TP09/tp9_data/tp9_data_train.txt', names=['x1', 'x2', 'y'])
df_valid = pd.read_csv('TP09/tp9_data/tp9_data_valid.txt', names=['x1', 'x2', 'y'])

# Calcul des paramètres pour chaque classe
def calcul_parametre(df):
    params = []
    for classe in range(5):
        X_classe = df[df['y'] == classe]
        moyenne = X_classe[['x1', 'x2']].mean().values
        cov = np.cov(X_classe[['x1', 'x2']].values.T)
        invCov = np.linalg.inv(cov)
        p = len(X_classe) / len(df)
        params.append([moyenne, invCov, p])
    return params

params = calcul_parametre(df_train)

# Fonction de prédiction basée sur la distance de Mahalanobis
def predictionMahanlobi(x, params):
    dist = []
    for param in params:
        delta = x - param[0]
        distance = delta.T @ param[1] @ delta - 2 * np.log(param[2])
        dist.append(distance)
    return np.argmin(dist)

# Préparation pour la visualisation
def plot_decision_multi(x1_min, x1_max, x2_min, x2_max, prediction, params, sample=300):
    x1_range = np.linspace(x1_min, x1_max, sample)
    x2_range = np.linspace(x2_min, x2_max, sample)
    y_pred = np.empty((sample, sample))
    for i, x1 in enumerate(x1_range):
        for j, x2 in enumerate(x2_range):
            y_pred[j, i] = prediction(np.array([x1, x2]), params)
    plt.contourf(x1_range, x2_range, y_pred, levels=5, alpha=0.35)

# Visualisation des données d'entraînement et des frontières de décision
plt.figure(figsize=(12, 8))
for label in range(5):
    subset = df_train[df_train['y'] == label]
    plt.scatter(subset['x1'], subset['x2'], label=label)
plot_decision_multi(df_train['x1'].min(), df_train['x1'].max(), df_train['x2'].min(), df_train['x2'].max(), predictionMahanlobi, params)
plt.legend()
plt.title('Frontières de décision avec la distance de Mahalanobis')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score

# Fonction de prédiction adaptée pour accepter les paramètres en argument
def predictionMahanlobi(x, params):
    dist = []
    for param in params:
        delta = x - param[0]
        distance = delta.T @ param[1] @ delta - 2 * np.log(param[2])
        dist.append(distance)
    return np.argmin(dist)

# Faire des prédictions sur l'ensemble de données de validation
y_pred = [predictionMahanlobi(row[['x1', 'x2']].values, params) for _, row in df_valid.iterrows()]

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(df_valid['y'], y_pred)

# Calculer le taux d'erreur
error_rate = 1 - accuracy_score(df_valid['y'], y_pred)

# Afficher les résultats
print("Matrice de confusion :")
print(conf_matrix)
print(f"Taux d'erreur : {error_rate * 100:.2f}%")
