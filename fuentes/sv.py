import numpy as np
import myutility as mt

def inf_gain(X,y):
    """
    Calculo de Ganancia de Informacion.
    """
    cols = X.shape[0]
    rows = X.shape[1]

    #Entropía de y; O entropia de clase
    I = 0
    valores_y, ocurrencias_y = np.unique(y,return_counts=True)
    #print(valores_y,ocurrencias_y,len(y))
    for i in range(valores_y.size):
        pi = ocurrencias_y[i] / len(y)
        #print(pi, np.log(pi))
        I -= (pi * np.log(pi))
        #print(I)
    print("Entropia variable objetivo",I)
    
    #Entropía ponderada del atributo
    E = []
    for j in range(cols):
        feature_entropy = calculateEntropy(X[j,:],y)
        E.append(feature_entropy)

    print(E)
    IG = I-E

    #Retorna un Arreglo de Ganancia de Informacion.
    return IG

def calculateEntropy(x,y):
    """
    Calculo de Entropia ponderada por columna.
    """

    x = np.array(x)
    n = x.shape[0] #N numero de muestras

    n_particiones = (np.ceil(np.sqrt(n))).astype(int) # numero de particiones

    particiones = np.linspace(min(x),max(x),num = n_particiones)
    #print(particiones,n_particiones)
    I = 0

    for i in range(n_particiones-1):
        valores_y = np.unique(y)#se extraen los valores posibles de y
        pi = np.zeros(len(valores_y))# se inicializa un arreglo para hacer conteo de instancias de cada valor de y

        for j in range(len(valores_y)): #se hace conteo de valores de i
            for k in range(len(x)): #se recorre el atributo
                if (particiones[i] <= x[k]) & (x[k] < particiones[i+1]): #si x está dentro de la partición
                    if(y[k] == valores_y[j]): #si es igual al valor y en que se itera
                        pi[j] += 1 #se suma uno 

        #calculo entropía de la particion
        if(pi.sum() != 0): #se asegura que no haya 0 instancias en la partición
            E = pi.sum()/n #se obtiene la ponderación del largo de la partición
            pi = pi/(pi.sum()) #se obtienen las probabilidades de los posibles valores de y en la partición
            partI = 0
            for valor_y in range(len(valores_y)): #se calcula la entropía de la partición
                if pi[valor_y] == 0:
                    continue
                partI -= pi[valor_y] * np.log(pi[valor_y])
            I += E*partI #se pondera y se suma a la entropía ponderada total

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

    #print(IG)
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

