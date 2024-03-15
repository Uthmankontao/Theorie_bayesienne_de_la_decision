"""def prediction_(x):
    delta = 147
    if delta > x:
        return("Classe 0: ")
    else:
        return("Classe 1: ")
for i in x_train :
    print(prediction_(i), i)"""


"""x = X_valid[y_valid==0]
    y = X_valid[y_valid==1]
    moyen_0 = x.mean()
    moyen_1 = y.mean()

    decision_f = (moyen_0 + moyen_1) / 2"""

import pandas as pd
import numpy as np

def Ultimate(data_train, data_valid):

    df_train = pd.read_csv(data_train, names=["X", "Y"])
    df_valid = pd.read_csv(data_valid, names=["X", "Y"])

    x_train = np.array(df_train["X"].values)
    y_train = np.array(df_train["Y"].values)
    x_valid = np.array(df_valid["X"].values)
    y_valid = np.array(df_valid["Y"].values)

    moyenne = np.mean(x_train)
    ecart = np.std(x_train)
    D_min = moyenne - ecart
    D_max = moyenne + ecart
    delta = np.arange(D_min, D_max + 0.1, 0.5)

    for i in delta:
        print("Avec delta = ", i)
        # Mise à jour de Frontiere pour utiliser le seuil i
        y_pred = np.where((x_valid >= i) & (x_valid <= i + 0.5), 1, 0)

        nbre_erreurs = np.sum(y_pred != y_valid)
        taux_erreurs = (nbre_erreurs / len(y_valid)) * 100

        print("le nombre d'erreurs: ", nbre_erreurs)
        print("avec un taux d'erreur de:", taux_erreurs, "%")

        TP = np.sum((y_pred == 0) & (y_valid == 0))
        FN = np.sum((y_pred == 1) & (y_valid == 0))
        FP = np.sum((y_pred == 0) & (y_valid == 1))
        TN = np.sum((y_pred == 1) & (y_valid == 1))
        matrice_confusion = np.array([[TP, FP], [FN, TN]])

        print("Matrice de confusion :")
        print(matrice_confusion)
Ultimate(data_train="Travaux pratiques/TP01/tp1_data/tp1_data_train.txt", data_valid="Travaux pratiques/TP01/tp1_data/tp1_data_valid.txt")