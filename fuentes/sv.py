import numpy as np
import myutility as mt

def inf_gain(X,y):
    cols = X.shape[0]
    rows = X.shape[1]
    #Entropía de y
    I = 0
    valores_y, ocurrencias_y = np.unique(y,return_counts=True)
    for i in range(valores_y.size):
        pi = ocurrencias_y[i]/len(y)
        I -= (pi*np.log2(pi))
        
    #Entropía ponderada del atributo
    E = []
    for j in cols:
        feature_entropy = 0
        
        for i in rows:
            
        E.append(feature_entropy)
    
    return I

'''
# SVD of X based on ppt
def svd_x(x):
    d = x.shape[0]
    N = x.shape[1]
    
    x_mean = np.zeros(x.shape)
    
    for i in range(0,d):
        x_mean[i] = x[i] - np.mean(x[i])
    x = x_mean
    #print(x)

    y = np.transpose(x) / np.sqrt(N - 1)
    #print(y, y.shape)
    u, s, v = np.linalg.svd(y)
    
    return(v)

'''