import myutility as mu
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def entropia(X):
    entropy_list = []
    #print("X SHAPE",X.shape[1])
    for variable in range(0,X.shape[1]):
      sumatoria = 0
      #print("set(X[:,variable])",set(X[:,variable]))
      values_cont = set(X[:,variable])
      for x in values_cont:
          d_i = np.shape(np.where(X[:,variable]==x))[1]
          
          p_i = d_i/X.shape[0]
          sumatoria = sumatoria + -(p_i*np.log2(p_i))
        
      entropy_list.append(sumatoria)
    return entropy_list


X, Y = mu.load_data("fuentes\KDDTrain.txt",0)
'''
res = mutual_info_classif(np.transpose(X), Y, discrete_features=False)
print(res)
'''

print(entropia(X))

#calculo de entropia para variables discretas
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
    
    #calculo de entropía para variables continuas
'''

   
    #====Separacion por Clase==============================================#
    
    ##NORMAL
    #if(config[4] == 0):
    #    aux = []
    #    #print("CLASE SELECCIONADA ELIMINADA: NORMAL")
    #    for i in range(db.shape[0]):
    #        if resultados.get(db[i,41]) == 1:
    #            aux.append(i)
    #    db = np.delete(db,aux,0) 
    #
#
    ##DOS
    #if(config[5] == 0):
    #    aux = []
    #    #print("CLASE SELECCIONADA ELIMINADA: DOS")
    #    for i in range(db.shape[0]):
    #        if resultados.get(db[i,41]) == 2:
    #            aux.append(i)
    #    db = np.delete(db,aux,0)  
    #
    ##Probe
    #if(config[6] == 0):
    #    aux = []
    #    #print("CLASE SELECCIONADA ELIMINADA: Probe")
    #    for i in range(db.shape[0]):
    #        if resultados.get(db[i,41]) == 3:
    #            aux.append(i)
    #    db = np.delete(db,aux,0)  

      #====================================TRAIN=============================#
    
    ##Selecciona el numero de filas de Train segun el config.
    #if(type==0):
    #    aux = random.sample(range(db.shape[0]), config[0])
    #    aux.sort()
    #    db = db[aux,:]
    #
    #
    ##=================================TEST=================================#
    ##Selecciona el numero de filas de Train segun el config.
    #if (type == 1):
    #    aux = random.sample(range(db.shape[0]), config[1])
    #    aux.sort()
    #    db = db[aux,:]


    #======================================================================#