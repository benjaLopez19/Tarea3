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
    print("Entropia variable objetivo",I)

    #Entropía ponderada del atributo
    E = []
    for j in range(cols):
        feature_entropy = calculateEntropy(X[j,:],y)

        E.append(feature_entropy)
    IG = I-E
    
    return IG

def calculateEntropy(x,y): #aca es donde muere y X recibe los 20000 registros en vez de los capeados.
    x = np.array(x)
    n = x.shape[0]
    I = 0
    valores_x, ocurrencias_x = np.unique(x, return_counts=True)
    #print(valores_x)
    count = []
    #arma diccionarios con el conteo de valores que tiene cada particion respecto de las variables objetivo
    for i in range(valores_x.shape[0]):
        count.append({})
        for j in range(n):
            if(x[j] == valores_x[i]):
                if y[j] in count[i]:
                    count[i][y[j]] = count[i][y[j]]+1
                else:
                    count[i][y[j]]=1 
        #print(count[i])

    
    for i in range(valores_x.shape[0]):
        E = ocurrencias_x[i]/n
        partI = 0
        for clase in count[i]:
            pi = count[i][clase]/ocurrencias_x[i]
            partI -= pi*np.log2(pi)
        I += E*partI

    #print(I)
    return I

# SVD of X based on ppt
def svd_x(x):
    d = x.shape[0]
    N = x.shape[1]
    
    x_mean = np.zeros(x.shape)
    for i in range(0,d):
        x_mean[i] = x[i] - np.mean(x[i])
    x = x_mean
    print("SHAPE CULIAO X",x.shape)

    y = np.transpose(x) / np.sqrt(N - 1)
    #print(y, y.shape)
    u, s, v = np.linalg.svd(y)
    #20000,40 - (40,40)
    return(v)

def select_variables():
    param = mt.load_config_sv()
    x,y = mt.load_data('fuentes\KDDTrain.txt',0)  
    
    relevancia = param[2]
    vectores_singulares = param[3]

    idx = []
    IG = inf_gain(x,y)
    for i in range(len(IG)):
        if IG[i] > relevancia:
           idx.append(i)
    print("xselect",x.shape)
    print("IDX",idx)
    x = x[idx,:]
    
    v = svd_x(x)
    if v.shape[1] > vectores_singulares:
        v = v[:,0:vectores_singulares]

    x = np.dot(np.transpose(v),x)   
    
    mt.save_filter(idx,v)

    return(x,y)


