from data_train import *

def prediction(x):
    delta = 145
    return np.where(x < delta, 0, 1)

# phase de validation
df_valid = pd.read_csv("Travaux pratiques/TP01/tp1_data/tp1_data_valid.txt", names= ["X", "Y"])

X_valid = np.array(df_valid["X"].values)
y_pred = prediction(X_valid)

# Calcul du nombre d'erreurs    
y_valid = np.array(df_valid["Y"].values)
nbre_erreurs = np.sum(y_pred != y_valid)
nbre_valid = len(y_valid)
taux_erreurs = (nbre_erreurs / nbre_valid)*100


def matrice_de_confusion(y_true, y_pred):
    TP = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 1) & (y_true == 0))
    FP = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 1) & (y_true == 1))
    return np.array([[TP, FP], [FN, TN]])


matrice_confusion = matrice_de_confusion(y_valid, y_pred)
print("Matrice de confusion :")
print(matrice_confusion)
print("le nombre d'erreurs: ", nbre_erreurs)
print("avec un taux d'erreur de:", taux_erreurs, "%")