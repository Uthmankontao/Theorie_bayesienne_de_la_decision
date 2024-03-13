import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def show(data_train):

    df = pd.read_csv(data_train, names=["X", "Y"])
    
    effectif_classe_0 = (df["Y"] == 0).sum()
    effectif_classe_1 = (df["Y"] == 1).sum()
    
    x_classe0 = df[df["Y"] == 0]["X"].values
    x_classe1 = df[df["Y"] == 1]["X"].values
    
    print(f"Effectifs de chaque classe:\nClasse 0: {effectif_classe_0}\nClasse 1: {effectif_classe_1}")
    

    plt.figure(figsize=(10, 6))
    plt.hist(x_classe0, bins=50, alpha=0.5, label="Classe 0", color="Blue")
    plt.hist(x_classe1, bins=50, alpha=0.5, label="Classe 1", color="Red")
    plt.xlabel("Valeurs des caractéristiques")
    plt.ylabel("Effectifs")
    plt.title("Histogramme de répartition des caractéristiques")
    plt.legend()
    plt.show()

data_train1 = "Travaux pratiques/TP02/tp2_data/tp2_data1_train.txt"
data_train2 = "Travaux pratiques/TP02/tp2_data/tp2_data2_train.txt"
data_train3 = "Travaux pratiques/TP02/tp2_data/tp2_data3_train.txt"



def test(data_valid):

    df_valid = pd.read_csv(data_valid, names=["X", "Y"])
    
    X_valid = np.array(df_valid["X"].values)
    y_valid = np.array(df_valid["Y"].values)
    
    x = X_valid[y_valid==0]
    y = X_valid[y_valid==1]
    moyen_0 = x.mean()
    moyen_1 = y.mean()

    decision_f = (moyen_0 + moyen_1) / 2

    y_pred = np.where(X_valid < decision_f, 0, 1)
    
    errors = np.sum(y_pred != y_valid)
    total = len(y_valid)
    error_rate = errors / total
    conf_matrix = confusion_matrix(y_valid, y_pred)
    print(decision_f)
    return error_rate, conf_matrix
    

data_valid1 = ("Travaux pratiques/TP02/tp2_data/tp2_data1_valid.txt")
data_valid2 = ("Travaux pratiques/TP02/tp2_data/tp2_data2_valid.txt")
data_valid3 = ("Travaux pratiques/TP02/tp2_data/tp2_data3_valid.txt")
error_rate, conf_matrix = test(data_valid3)

print(error_rate)
print(conf_matrix)

show(data_train3)