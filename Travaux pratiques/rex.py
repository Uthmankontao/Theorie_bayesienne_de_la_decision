import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Pour calculer les paramètres de l'estimateur de maximum de vraisemblance sous l'hypothèse de gaussiennes isotropes pour votre ensemble 
de données, nous devons suivre plusieurs étapes. Ces étapes impliquent le calcul des moyennes et des variances de vos données, séparément
pour chaque classe si votre problème est de classification. Dans le cas des gaussiennes isotropes, la matrice de covariance est supposée
être la même pour toutes les classes et proportionnelle à la matrice identité, ce qui simplifie certains calculs.
L'hypothèse de gaussiennes isotropes signifie que les covariances des distributions gaussiennes sont des multiples de la matrice identité.

Les étapes sont généralement les suivantes :
Calculer la moyenne de chaque classe.
Calculer la variance globale (sous l'hypothèse isotrope, la matrice de covariance pour chaque classe est supposée être un scalaire (la variance)
multiplié par la matrice identité).
Utiliser ces paramètres pour classifier de nouvelles observations ou pour estimer la densité de probabilité des observations existantes
sous ce modèle.

"""

df_train = pd.read_csv("Travaux pratiques/TP05/tp5_data/tp5_data1_train.txt", names=["x1", "x2", "y"])
X_train = df_train[["x1", "x2"]].values
y_train = df_train["y"].values

# Je vais d'abord calculer la moyenne pour chaque classe:

classe0 = df_train[df_train["y"]==0]
classe1 = df_train[df_train["y"]==1]

u0 = classe0[["x1","x2"]].mean()
u1 = classe1[["x1","x2"]].mean()
print(f'La moyenne des caractéristiques de la classe 0:\n {u0}')
print(f'La moyenne des caractéristiques de la classe 1:\n {u1}')

# Je vais calculer la variance globale (isotrope) pour l'ensemble des données
# La variance isotrope est calculée comme la moyenne des variances de x1 et x2 sur l'ensemble des données
#Calculer la variance commune variance isotrope globale à partir des distances euclidiennes des points de chaque classe par rapport à leur moyenne respective.

sigma2 = df_train[['x1', 'x2']].var().mean()
print(f'La variance globale isotrope est: {sigma2}')

def pdf_gaussian(x, u, sigma2):
    """
    Calcul de la fonction de densité de probabilité (PDF) d'une distribution gaussienne.
    """
    return (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - u) ** 2) / (2 * sigma2))

def prediction(x):
    """
    Prédiction de la classe d'un vecteur x en utilisant la règle de Bayes sous l'hypothèse de gaussiennes isotropes.
    """
    # Probabilités a priori des classes (estimées par les fréquences relatives des classes)
    p_c0 = len(classe0) / len(df_train)
    p_c1 = len(classe1) / len(df_train)
    
    # Calcul de la probabilité de x pour chaque classe
    p_x_sachant_c0 = pdf_gaussian(x[0], u0['x1'], sigma2) * pdf_gaussian(x[1], u0['x2'], sigma2) * p_c0
    p_x_sachant_c1 = pdf_gaussian(x[0], u1['x1'], sigma2) * pdf_gaussian(x[1], u1['x2'], sigma2) * p_c1
    
    # Retourne la classe avec la probabilité postérieure la plus élevée
    return 0 if p_x_sachant_c0 > p_x_sachant_c1 else 1
















