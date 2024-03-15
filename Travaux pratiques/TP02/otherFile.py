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
delta_tp1 = 145
print("RESULTATS DES TESTS DU TP01")
print("////////////////////////////////////////////////////////////////////////////////////////////////////////")
test(delta_tp1, data_valid_tp1)
print("////////////////////////////////////////////////////////////////////////////////////////////////////////\n")



# Les tests pour le TP 2
delta_1 = 56
delta_2 = 456
delta_3 = 330
print("RESULTATS DES TESTS DU TP02")
print("////////////////////////////////////////////////////////////////////////////////////////////////////////")
print("DATA_VALID_1")
test(delta_1, data_valid1)
print("////////////////////////////////////////////////////////////////////////////////////////////////////////")
print("DATA_VALID_2")
test(delta_2, data_valid2)
print("////////////////////////////////////////////////////////////////////////////////////////////////////////")
print("DATA_VALID_3")
test(delta_3, data_valid3)
print("////////////////////////////////////////////////////////////////////////////////////////////////////////")
