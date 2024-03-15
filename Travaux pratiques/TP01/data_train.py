import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Travaux pratiques/TP01/tp1_data/tp1_data_train.txt', names= ["X", "Y"])
print("Dataframe", df)

# Conversion en tableau avec numpy
x_train = np.array(df["X"].values) # ma matrice avec les caractéristiques de X
y_train = np.array(df["Y"].values) # ma matrice contenant les valeurs des classes

# separation des données 
x_classe0 = x_train[y_train==0]
x_classe1 = x_train[y_train==1]

# Affichage des histogrammes
plt.figure(figsize=(10,6))  
plt.hist(x_classe0, bins=50, alpha=0.5, label="Classe 0", color="Blue")
plt.hist(x_classe1, bins=50, alpha = 0.5, label="Classe 1", color="Red")
plt.xlabel("Valeurs des caractéristiques")
plt.ylabel("Effectifs")
plt.title("Histogramme de repartition des caractéristiques: TP1_data")
plt.legend()
plt.show()
# Visuellement je prendrai un delta = 147
#Si valeur > delta alors la classe est 1 sinon 0