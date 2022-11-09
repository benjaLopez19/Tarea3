import numpy as np
import myutility as mt

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
    n = x.shape[0] #N numero de muestras
 
    aux = []
    index =x.argsort()
    for i in index:
        aux.append(y[i])
    y = aux

    n_particiones = (np.ceil(np.sqrt(n))).astype(int)
    I = 0

    #Se obtienen las particiones del atributo
    particiones = np.array_split(x,n_particiones)
    particiones = np.array(particiones)
    y_particiones = np.array_split(y,n_particiones)
    y_particiones = np.array(y_particiones)

    for i in range(n_particiones): #Se itera en el número de particiones del atributo
        #se obtiene el conteo de posibles valores objetivo de la i-ésima partición
        valores_y = np.unique(y_particiones[i])#se extraen los valores posibles de y
        pi = np.zeros(len(valores_y))# se inicializa un arreglo para hacer conteo de instancias de cada valor de y
        for j in range(len(valores_y)):#se itera en los valores de y
            for k in range(len(y_particiones[i])):#se itera sobre el largo de la partición actual
                if y_particiones[i][k] == valores_y[j]:#si el valor actual de la partición actual es igual al posible valor de y sobre el que se itera
                    pi[j] += 1 #se añade uno al contador
        #Cálculo de la entropía de la partición
        E = len(particiones[i])/n #se obtiene la ponderación del largo de la partición
        pi = pi/(len(particiones[i])) #se obtienen las probabilidades de los posibles valores de y en la partición
        partI = 0
        for valor_y in range(len(valores_y)): #se calcúla la entropía de la partición
            partI -= pi[valor_y] * np.log2(pi[valor_y])
        I += E*partI #se pondera y se suma a la entropía ponderada total
    #print("{:.5f}".format(I))
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

    #sort
    IG = inf_gain(x,y)

    index = IG.argsort()
    index = np.flip(index,0)

    index = index[:relevancia]

    #print("xselect",x.shape)
    #print("IDX",idx)
    x = x[index,:]
    #print(x.shape)
    v = svd_x(x)
    #Se cortan todos los datos que sean mayores a los vectores singulares.
    if v.shape[1] > vectores_singulares:
        v = v[:,0:vectores_singulares]

    x = np.dot(np.transpose(v),x)   
    mt.save_filter(index,v)

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

