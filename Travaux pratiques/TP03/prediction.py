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

    delta = np.linspace(D_min, D_max, 1000) 
    meilleur_taux_erreur = np.inf
    meilleure_frontiere = None

    for i in delta:
        y_pred = np.where(x_valid < i, 0, 1)
        taux_erreurs = np.mean(y_pred != y_valid) * 100

        if taux_erreurs < meilleur_taux_erreur:
            meilleur_taux_erreur = taux_erreurs
            meilleure_frontiere = i

    y_pred = np.where(x_valid < meilleure_frontiere, 0, 1)
    nbre_erreurs = np.sum(y_pred != y_valid)

    TP = np.sum((y_pred == 0) & (y_valid == 0))
    FN = np.sum((y_pred == 1) & (y_valid == 0))
    FP = np.sum((y_pred == 0) & (y_valid == 1))
    TN = np.sum((y_pred == 1) & (y_valid == 1))
    matrice_confusion = np.array([[TP, FP], [FN, TN]])

    print(f"Meilleure frontière de décision: {meilleure_frontiere}")
    print(f"Meilleur taux d'erreur: {meilleur_taux_erreur}%")
    print(f"Nombre d'erreur: {nbre_erreurs}")
    print("Matrice de confusion :")
    print(matrice_confusion)


############################################################################################################################################
                                                            #Test#
print("####################################################################################################################################")
print("DATA1\n")
Ultimate(data_train="Travaux pratiques/TP03/tp3_data/tp3_data1_train.txt", data_valid="Travaux pratiques/TP03/tp3_data/tp3_data1_valid.txt")
print("####################################################################################################################################")
print("DATA2\n")
Ultimate(data_train="Travaux pratiques/TP03/tp3_data/tp3_data2_train.txt", data_valid="Travaux pratiques/TP03/tp3_data/tp3_data2_valid.txt")
print("####################################################################################################################################")
print("DATA3\n")
Ultimate(data_train="Travaux pratiques/TP03/tp3_data/tp3_data3_train.txt", data_valid="Travaux pratiques/TP03/tp3_data/tp3_data3_valid.txt")
print("####################################################################################################################################")
print("DATA4\n")
Ultimate(data_train="Travaux pratiques/TP03/tp3_data/tp3_data4_train.txt", data_valid="Travaux pratiques/TP03/tp3_data/tp3_data4_valid.txt")
print("####################################################################################################################################")
print("DATA5\n")
Ultimate(data_train="Travaux pratiques/TP03/tp3_data/tp3_data5_train.txt", data_valid="Travaux pratiques/TP03/tp3_data/tp3_data5_valid.txt")
print("####################################################################################################################################")
print("DATA6\n")
Ultimate(data_train="Travaux pratiques/TP03/tp3_data/tp3_data6_train.txt", data_valid="Travaux pratiques/TP03/tp3_data/tp3_data6_valid.txt")
print("####################################################################################################################################")
print("DATA7\n")
Ultimate(data_train="Travaux pratiques/TP03/tp3_data/tp3_data7_train.txt", data_valid="Travaux pratiques/TP03/tp3_data/tp3_data7_valid.txt")
    








   



