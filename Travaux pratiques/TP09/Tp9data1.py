import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import utils_TP9 as ut

couleur={1:'blue',0:'orange',2:'red',3:'cyan',4:'marron'}
def calcul_parametre():
    param=[]
    df=pd.read_csv('TP09/tp9_data/tp9_data_train.txt',names=['x1','x2','y'])
    for classe in range(5):
            X_classe=df[df['y']==classe]
            X_classe=X_classe[['x1','x2']]
            moyenne=[X_classe['x1'].mean(),X_classe['x2'].mean()]
            cov=(X_classe-moyenne).T@(X_classe-moyenne)
            cov=cov/len(X_classe)
            detCov=np.linalg.det(cov)
            invCov=np.linalg.inv(cov)
            p=len(df[df['y']==classe])/len(df)
            param.append([moyenne,detCov,invCov,p])
    return param
param=calcul_parametre()
def predictionMahanlobi(x):
    dist=[]
    for i in range(5):
        dist.append((x-param[i][0]).T @ param[i][2] @ (x-param[i][0])+np.log(param[i][0][1])-2*np.log(param[i][3]))
    dist=np.array(dist)
    return np.argmin(dist)


ddf=pd.read_csv('TP09/tp9_data/tp9_data_train.txt',names=['x1','x2','y'])
X_valid=ddf[['x1','x2']]
Y_valid=np.array(ddf['y'])
couleur={1:'blue',0:'orange'}
plt.figure(figsize=(12,8))
for label in np.unique(Y_valid):
    plt.scatter(ddf[Y_valid == label]['x1'], ddf[Y_valid == label]['x2'], label=label, marker='+' if label == 0 else 'x')
plt.legend()

ut.plot_decision_multi(X_valid['x1'].min(),X_valid['x1'].max(),X_valid['x2'].min(),X_valid['x2'].max(),prediction=predictionMahanlobi)
plt.axis('equal')
plt.savefig('fig_TP9')
plt.show()























