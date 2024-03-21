from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def Ultimate(data_train, data_valid):

    df_train = pd.read_csv(data_train, names=["x1", "x2", "y"])
    X_train = df_train[["x1", "x2"]].values
    y_train = df_train["y"].values

    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    def prediction(x):
        return logistic_model.predict(x.reshape(1, -1))[0]

    def plot_decision(x1_min, x1_max, x2_min, x2_max, prediction, sample = 300):
        """Uses Matplotlib to plot and fill a region with 2 colors
        corresponding to 2 classes, separated by a decision boundary

        Parameters
        ----------
        x1_min : float
            Minimum value for the first feature
        x1_max : float
            Maximum value for the first feature
        x2_min : float
            Minimum value for the second feature
        x2_max : float
            Maximum value for the second feature
        prediction :  (x : 2D vector) -> label : int
            Prediction function for decision
        sample : int, optional
            Number of samples on each feature (default is 300)
        """
        x1_list = np.linspace(x1_min, x1_max, sample)
        x2_list = np.linspace(x2_min, x2_max, sample)
        y_grid_pred = [[prediction(np.array([x1,x2])) for x1 in x1_list] for x2 in x2_list] 
        plt.contourf(x1_list, x2_list, y_grid_pred, levels=1, colors=["blue","orange"],alpha=0.35)

    x1_min, x1_max = X_train[:, 0].min(), X_train[:, 0].max() 
    x2_min, x2_max = X_train[:, 1].min(), X_train[:, 1].max() 
    plt.figure(figsize=(8, 6))
    plot_decision(x1_min, x1_max, x2_min, x2_max, prediction)
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='blue', marker='+', label='Classe 0')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='orange', marker='x', label='Classe 1')
    plt.title('Frontière de décision linéaire')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    def Validation(data_valid):
        df_valid = pd.read_csv(data_valid, names=["x1", "x2", "y"])
        x_valid = df_valid[["x1", "x2"]].values
        y_valid = df_valid["y"].values

        y_pred = np.array([prediction(x) for x in x_valid])
        nbre_erreurs = np.sum(y_pred != y_valid)
        taux_erreur = np.mean(y_pred != y_valid) * 100
        def matrice_de_confusion(y_true, y_pred):
            TP = np.sum((y_pred == 0) & (y_true == 0))
            FN = np.sum((y_pred == 1) & (y_true == 0))
            FP = np.sum((y_pred == 0) & (y_true == 1))
            TN = np.sum((y_pred == 1) & (y_true == 1))
            return np.array([[TP, FP], [FN, TN]])
        conf_matrix_custom = matrice_de_confusion(y_valid, y_pred)

        print(f'Le nombre d\'erreur est {nbre_erreurs} soit un taux d\'erreur de {taux_erreur} %')
        print(conf_matrix_custom)

    Validation(data_valid)

print("Les resultats des données 1")
Ultimate(data_train="Travaux pratiques/TP05/tp5_data/tp5_data1_train.txt", data_valid="Travaux pratiques/TP05/tp5_data/tp5_data1_valid.txt")
print("{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}{|}")
print("Les resultats des données 2")
Ultimate(data_train="Travaux pratiques/TP05/tp5_data/tp5_data2_train.txt", data_valid="Travaux pratiques/TP05/tp5_data/tp5_data2_valid.txt")