from tools import show, test
data_valid_tp1 = "Travaux pratiques/TP01/tp1_data/tp1_data_valid.txt"
data_valid1 = "Travaux pratiques/TP02/tp2_data/tp2_data1_valid.txt"
data_valid2 = "Travaux pratiques/TP02/tp2_data/tp2_data2_valid.txt"
data_valid3 = "Travaux pratiques/TP02/tp2_data/tp2_data3_valid.txt"

data_train_tp1 = "Travaux pratiques/TP01/tp1_data/tp1_data_train.txt"
data_train1 = "Travaux pratiques/TP02/tp2_data/tp2_data1_train.txt"
data_train2 = "Travaux pratiques/TP02/tp2_data/tp2_data2_train.txt"
data_train3 = "Travaux pratiques/TP02/tp2_data/tp2_data3_train.txt"

# Les tests pour le TP 1
print("RESULTATS DES TESTS DU TP01")
test(data_valid_tp1)


# Les tests pour le TP 2
print("RESULTATS DES TESTS DU TP02")
test(data_valid1)
test(data_valid2)
test(data_valid3)
