import numpy as np
import myutility as mt
import math

def inf_gain(X,y):
    """
    Calculo de Ganancia de Informacion.
    """
    cols = X.shape[0]
    rows = X.shape[1]

    #Entropía de y o entropia de clase
    I = 0
    valores_y, ocurrencias_y = np.unique(y,return_counts=True)
    for i in range(valores_y.size):
        pi = ocurrencias_y[i] / len(y)
        I -= (pi * np.log2(pi))
    #print("Entropia variable objetivo",I)
    
    #Entropía ponderada del atributo
    E = []
    for j in range(cols):
        feature_entropy = calculateEntropy(X[j,:],y)
        E.append(feature_entropy)

    IG = I-E

    #Retorna un Arreglo de Ganancia de Informacion.
    return IG

def calculateEntropy(x,y):
    """
    Calculo de Entropia ponderada por columna.
    """

    x = np.array(x)
    n = x.shape[0]
    print("X",x)
    
    n_particiones = math.ceil(math.sqrt(n))
    print("n_particiones", n_particiones)
    print("n",n)
    I = 0

    #Valores individuales
    particiones = np.array_split(x,n_particiones)
    particiones = np.array(particiones)
    y_particiones = np.array_split(y,n_particiones)
    y_particiones = np.array(y_particiones)

    #print("Dimension x",particiones.shape,"Dimension y",y_particiones.shape)
    #print(particiones,y_particiones)

    '''
    valores_x, ocurrencias_x = np.unique(x, return_counts=True)
    
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

    #Se calcula la sumatoria para la entropia de valores individuales
    for i in range(valores_x.shape[0]):
        E = ocurrencias_x[i]/n #largo de la particion/ n
        partI = 0
        for clase in count[i]:
            pi = count[i][clase]/ocurrencias_x[i]
            partI -= pi * np.log2(pi)
        I += E*partI
    '''
    for i in range(n_particiones):
        #se obtiene el conteo de posibles valores objetivo de la i-ésima partición
        valores_y = np.unique(y_particiones[i])
        pi = np.zeros(len(valores_y))
        for j in range(len(valores_y)):
            for k in range(len(y_particiones[i])):
                if y_particiones[i][k] == valores_y[j]:
                    pi[j] += 1
        #se obtiene la probabilidad de clase
        E = len(particiones[i])/n
        pi = pi/(len(particiones[i]))
        partI = 0
        for valor_y in range(len(valores_y)):
            partI -= pi[valor_y] * np.log2(pi[valor_y])
        I += E*partI
    print("I",I)
    return I

def svd_x(x):
    """
    Funcion de SVD.
    """
    d = x.shape[0]
    N = x.shape[1]
    
    x_mean = np.zeros(x.shape)
    for i in range(0,d):
        x_mean[i] = x[i] - np.mean(x[i])
    x = x_mean

    y = np.transpose(x) / np.sqrt(N - 1)

    u, s, v = np.linalg.svd(y)
    
    return(v)

def select_variables():
    """
    Seleccion de variables que utiliza ganancia de información y reducción de redunancia con SVD.
    """
    param = mt.load_config_sv()
    x,y = mt.load_data('fuentes\KDDTrain.txt',0) 

    relevancia = param[2]
    vectores_singulares = param[3]

    idx = []

    #sort
    aux = [] 
    IG = inf_gain(x,y)
    for i in (range(len(IG))):
        aux.append([IG[i],i] #appendea a un auxiliar valor de IG 
        )
    #print(aux)
    #print("")
    aux = np.array(aux)
    #print("")
    #print(aux)
    aux = aux[aux[:,0].argsort()]
    #print("")
    #print(aux)
    ##################
    aux = np.flip(aux,0)
    #print(aux)
    index = aux[:,1].astype(int) #mis nuevos índices, se supone, porque no sé si el sort está bien
    print("index",index)

    for i in range(len(IG)):
        if IG[i] > relevancia:
           idx.append(i)
    #print("xselect",x.shape)
    #print("IDX",idx)
    x = x[idx,:]
    
    v = svd_x(x)
    #Se cortan todos los datos que sean mayores a los vectores singulares.
    if v.shape[1] > vectores_singulares:
        v = v[:,0:vectores_singulares]

    x = np.dot(np.transpose(v),x)   
    mt.save_filter(idx,v)

    return(x,y)

def main():
    x,y = select_variables()
    np.savetxt('dtrn.csv', x, fmt='%d', header=' ',  delimiter=' , ')
    np.savetxt('etrn.csv', y, fmt='%d', header=' ',  delimiter=' , ')

    #sacar archivo de indices
    index = np.loadtxt('index_var.csv', dtype= int, delimiter=";")

    #sacar archivo de matriz v
    v = np.loadtxt('filter.csv', dtype= int, delimiter=";")

    #cargar datos de testeo
    x_test , y_test = mt.load_data('fuentes/KDDTest.txt',1)

    #filtrar/caracteristicas filas por indice
    x_test = x_test[index,:]
    #filtrar con v (vtraspuesta*x)
    x_dtst = np.dot(np.transpose(v),x_test)   

    #crear x como dtst.csv
    np.savetxt('dtst.csv', x_dtst, fmt='%d', header=' ',  delimiter=' , ')
    #guardar y como etst.csv
    np.savetxt('etst.csv', y_test, fmt='%d', header=' ',  delimiter=' , ')


if __name__ == '__main__':   
	 main()

