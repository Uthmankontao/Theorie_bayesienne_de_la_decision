import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def show(data_train):

    df = pd.read_csv(data_train, names=["X", "Y"])
    
    effectif_classe_0 = (df["Y"] == 0).sum()
    effectif_classe_1 = (df["Y"] == 1).sum()
    print(f"Effectifs de chaque classe:\nClasse 0: {effectif_classe_0}\nClasse 1: {effectif_classe_1}")
    
    x_classe0 = df[df["Y"] == 0]["X"].values
    x_classe1 = df[df["Y"] == 1]["X"].values   
    plt.figure(figsize=(10, 6))
    plt.hist(x_classe0, bins=50, alpha=0.5, label="Classe 0", color="Blue")
    plt.hist(x_classe1, bins=50, alpha=0.5, label="Classe 1", color="Red")
    plt.xlabel("Valeurs des caractéristiques")
    plt.ylabel("Effectifs")
    plt.title("Histogramme de répartition des caractéristiques")
    plt.legend()
    plt.show()



def test(data_valid):

    df_valid = pd.read_csv(data_valid, names=["X", "Y"])
    
    X_valid = np.array(df_valid["X"].values)
    y_valid = np.array(df_valid["Y"].values)
    
    def prediction(x):
        delta = X_valid.mean()
        print("Frontière de décision: ", delta)
        return np.where(x < delta, 0, 1)

    y_pred = prediction(X_valid)
    nbre_erreurs = np.sum(y_pred != y_valid)
    nbre_valid = len(y_valid)
    taux_erreurs = (nbre_erreurs / nbre_valid) *100
    print("le nombre d'erreurs: ", nbre_erreurs)
    print("avec un taux d'erreur de:", taux_erreurs, "%")

    def matrice_de_confusion(y_true, y_pred):
        TP = np.sum((y_pred == 0) & (y_true == 0))
        FN = np.sum((y_pred == 1) & (y_true == 0))
        FP = np.sum((y_pred == 0) & (y_true == 1))
        TN = np.sum((y_pred == 1) & (y_true == 1))
        return np.array([[TP, FP], [FN, TN]])
    matrice_confusion = matrice_de_confusion(y_valid, y_pred)
    print("Matrice de confusion :")
    print(matrice_confusion)



# test de ma fonction:
