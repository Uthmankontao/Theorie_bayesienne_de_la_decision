
# Visuellement je prendrai un delta = 147

#Si valeur > delta alors la classe est 1 sinon 0

def prediction(x):
    
    delta = 147
    if delta > x:
        return("Classe 0")
    else:
        return("Classe 1")



for i in x_train : 