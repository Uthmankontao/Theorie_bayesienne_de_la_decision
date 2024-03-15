import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Visualisation(data_train):

    df_train = pd.read_csv(data_train, names=["X", "Y"])

    x_train = np.array(df_train["X"].values)
    y_train = np.array(df_train["Y"].values)

    classe_0 = x_train[y_train == 0]
    classe_1 = x_train[y_train == 1]
    classe_2 = x_train[y_train == 2]

    plt.figure(figsize=(10,6))  
    plt.hist(classe_0, bins=50, alpha=0.5, label="Classe 0", color="green")
    plt.hist(classe_1, bins=50, alpha = 0.5, label="Classe 1", color="blue")
    plt.hist(classe_2, bins=50, alpha = 0.5, label="Classe 2", color="Red")
    plt.xlabel("Valeurs des caractéristiques")
    plt.ylabel("Effectifs")
    plt.title("Histogramme de repartition des caractéristiques")
    plt.legend()
    plt.show()

print("####################################################################################################################################")
print("DATA1\n")
Visualisation(data_train="Travaux pratiques/TP04/tp4_data/tp4_data1_train.txt")
print("####################################################################################################################################")
print("DATA2\n")
Visualisation(data_train="Travaux pratiques/TP04/tp4_data/tp4_data2_train.txt")
    
def Ultimate(data_train, data_valid):
    df_train = pd.read_csv(data_train, names=["X", "Y"])
    df_valid = pd.read_csv(data_valid, names=["X", "Y"])

    x_train = np.array(df_train["X"].values)
    x_valid = np.array(df_valid["X"].values)
    y_valid = np.array(df_valid["Y"].values)

    moyenne = np.mean(x_train)
    ecart = np.std(x_train)
    D_min = moyenne - ecart
    D_max = moyenne + ecart

    delta = np.linspace(D_min, D_max, 100)
    meilleur_taux_erreur = np.inf
    meilleurs_seuils = (None, None)

    def predire_classe(x, seuil_1, seuil_2):
        if x < seuil_1:
            return 0
        elif x < seuil_2:
            return 1
        else:
            return 2
    
    for seuil_1 in delta:
        for seuil_2 in delta:
            if seuil_1 < seuil_2:  
                y_pred = [predire_classe(x, seuil_1, seuil_2) for x in x_valid]
                taux_erreurs = np.mean(y_pred != y_valid) * 100

                if taux_erreurs < meilleur_taux_erreur:
                    meilleur_taux_erreur = taux_erreurs
                    meilleurs_seuils = (seuil_1, seuil_2)

    y_pred = [predire_classe(x, meilleurs_seuils[0], meilleurs_seuils[1]) for x in x_valid]
    def calculer_matrice_de_confusion(y_true, y_pred):
        classes = np.unique(y_true)
        matrice = np.zeros((len(classes), len(classes)), dtype=int)
        for i, classe_vraie in enumerate(classes):
            for j, classe_predite in enumerate(classes):
                matrice[i, j] = np.sum((y_true == classe_vraie) & (y_pred == classe_predite))
        return matrice
    y_pred_finale = [predire_classe(x, meilleurs_seuils[0], meilleurs_seuils[1]) for x in x_valid]
    matrice_confusion = calculer_matrice_de_confusion(y_valid, y_pred_finale)
    print(f"Meilleurs seuils: {meilleurs_seuils}")
    print(f"Meilleur taux d'erreur: {meilleur_taux_erreur}%")
    print("Matrice de confusion :")
    print(matrice_confusion)

print("####################################################################################################################################")
print("DATA1\n")
Ultimate(data_train="Travaux pratiques/TP04/tp4_data/tp4_data1_train.txt", data_valid="Travaux pratiques/TP04/tp4_data/tp4_data1_valid.txt")
print("####################################################################################################################################")
print("DATA2\n")
Ultimate(data_train="Travaux pratiques/TP04/tp4_data/tp4_data2_train.txt", data_valid="Travaux pratiques/TP04/tp4_data/tp4_data2_valid.txt")