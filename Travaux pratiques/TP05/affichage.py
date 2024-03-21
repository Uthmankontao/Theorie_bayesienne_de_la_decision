import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Show(data_train):

    df_train = pd.read_csv("Travaux pratiques/TP05/tp5_data/tp5_data1_train.txt", names=["x1", "x2", "y"])

    X_train = df_train[["x1", "x2"]].values
    y_train = df_train["y"]

    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='blue', marker='+', label='Classe 0')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='orange', marker='x', label='Classe 1')

    plt.axis("equal")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Visualisation des données d'entraînement avec X_train et y_train")
    plt.legend()
    plt.show()



Show(data_train="Travaux pratiques/TP05/tp5_data/tp5_data1_train.txt")

