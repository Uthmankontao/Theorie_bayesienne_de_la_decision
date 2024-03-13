import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Création de la dataframe
df = pd.read_csv('Travaux pratiques/TP01/tp1_data/tp1_data_train.txt', names= ["X", "Y"])
print("Dataframe", df)


# Conversion en tableau avec numpy
x_train = np.array(df["X"].values) # ma matrice avec les caractéristiques de X
print("Matrice des caractéristiques:", x_train)


y_train = np.array(df["Y"].values) # ma matrice contenant les valeurs des classes
print("Matrice des classes:", y_train)

# Nombre de données dans les differentes classes
print("Il y'a ", (df["Y"] == 0).sum(), " données dans la classe 0")

print("Il y'a ", (df["Y"] == 1).sum(), " données dans la classe 1")

# Affichage des histogrammes 

# separation des données 
x_classe0 = x_train[y_train==0]
print(len(x_classe0))

x_classe1 = x_train[y_train==1]
print(len(x_classe1))

plt.figure(figsize=(10,6))

plt.hist(x_classe0, bins=50, alpha=0.5, label="Classe 0", color="Blue")
plt.hist(x_classe1, bins=50, alpha = 0.5, label="Classe 1", color="Red")

plt.xlabel("Valeurs des caractéristiques")
plt.ylabel("Effectifs")
plt.title("Histogramme de repartition des caractéristiques")
plt.legend()

#plt.show()





def prediction_(x):
    
    delta = 147
    if delta > x:
        return("Classe 0: ")
    else:
        return("Classe 1: ")
for i in x_train :
    print(prediction_(i), i)
    

def prediction(x):
    delta = 147
    return np.where(x < delta, 0, 1)

# phase de validation
    
df_valid = pd.read_csv("Travaux pratiques/TP01/tp1_data/tp1_data_valid.txt", names= ["X", "Y"])

X_valid = np.array(df["X"].values)

y_valid = np.array(df["Y"].values)

print(df_valid.head())


y_pred = prediction(X_valid)


errors = np.sum(y_pred != y_valid)
total = len(y_valid)
error_rate = errors / total

conf_matrix = confusion_matrix(y_valid, y_pred)

print(errors, error_rate, conf_matrix)
