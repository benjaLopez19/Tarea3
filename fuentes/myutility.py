import numpy as np
import time
import sv 
import os
import random

#Valores de columna objetivo
resultados = {  "normal":1,
                #DOS
                'neptune':2, 'teardrop':2, 'smurf':2, 'pod':2, 'back':2,
			    'land':2, 'apache2':2, 'processtable':2, 'mailbomb':2,'udpstorm':2,
                #Probe
                'ipsweep':3, 'portsweep':3, 'nmap':3,
                'satan':3, 'saint':3, 'mscan':3, 
                #?
                #"warezclient":4,"guess_passwd":4,"ftp_write":4,'multihop':4
            }

#LECTURA ARCHIVO

def load_config_sv():      
    param = np.genfromtxt("configs/cnf_sv.csv",delimiter=',',dtype=None)    
    par=[]    
    par.append(np.int16(param[0])) # N. muestrar train 
    par.append(np.int16(param[1])) # N muestras test
    par.append(np.float(param[2])) # Valor de relevancia
    par.append(np.int16(param[3])) # N vectores singulares  
    par.append(np.int16(param[4])) # Clase normal     
    par.append(np.int16(param[5])) # Clase DOS 
    par.append(np.int16(param[6])) # Clase Probe
    return(par)


def load_data(fname,type,clase):
    db  = np.loadtxt(fname, dtype= str, delimiter=",")
    index = []

    #reduccion de filas
    #Se eliminan los valores no especificados en la ppt para la lista de ataque
    for i in range(db.shape[0]):
        if resultados.get(db[i,41]) is None:
            index.append(i)
    db = np.delete(db,index,0)

    config = load_config_sv()

#====separacion por clase==============================================#
    
    #NORMAL
    if(config[4] == 0):
        aux = []
        print("CLASE SELECCIONADA ELIMINADA: NORMAL")
        for i in range(db.shape[0]):
            if resultados.get(db[i,41]) == 1:
                aux.append(i)
        db = np.delete(db,aux,0) 
    
    

    #DOS
    if(config[5] == 0):
        aux = []
        print("CLASE SELECCIONADA ELIMINADA: DOS")
        for i in range(db.shape[0]):
            if resultados.get(db[i,41]) == 2:
                aux.append(i)

        db = np.delete(db,aux,0)  
    
    #Probe
    if(config[6] == 0):
        aux = []
        print("CLASE SELECCIONADA ELIMINADA: Probe")
        for i in range(db.shape[0]):
            if resultados.get(db[i,41]) == 2:
                aux.append(i)

        db = np.delete(db,aux,0)  
    
                


#====================================TRAIN=============================#

    if(type==0):
        aux = random.sample(range(db.shape[0]), config[0])
        aux.sort()
        db = db[aux,:]
    
    
#=================================TEST=======================================#
    if (type == 1):
        aux = random.sample(range(db.shape[0]), config[1])
        aux.sort()
        db = db[aux,:]


#====================================================================================#

    aux_y = db[:,41] #Columna con valores objetivos
    y = []
    X = np.delete(db,41,1) #Resto de la base de datos

    #Se cambian valores de la columna objetivo a números
    for i in range(aux_y.size):
        y.append(resultados[aux_y[i]])
 
    
    #obtención de diccionarios para convertir variables no numéricas
    dicts = []
    aux = 0
    for j in [1,2,3]:
        dicts.append({})
        cont = 1
        for i in range(X.shape[0]):
            if dicts[aux].get(X[i,j]) is None:
                dicts[aux][str(X[i,j])] = cont
                cont += 1
        aux +=1
    #reemplazo de variables no numéricas
    aux = 0
    for j in [1,2,3]:
        for i in range(X.shape[0]):
            X[i,j] = dicts[aux][X[i,j]]
        aux +=1
    #se transforman los datos a flotante
    X = X.astype(float)
    X = normalizar(X)
    return (np.transpose(X),y)

#Guardado de filtro e indices relevantes
def save_filter(idx,V):
    idx_2 = np.asarray(idx)
    idx_2 = np.reshape(idx_2,(idx_2.shape[0],1))
    #print(idx_2)
    np.savetxt('index_var.csv', idx_2, fmt='%d', header=' ',  delimiter=' ; ')
    np.savetxt('filter.csv', V, fmt='%1.13f', header=' ',  delimiter=' ; ') 
    
    return()

#Normalización de datos
def normalizar(x):
    rows = x.shape[0]
    cols = x.shape[1]
    xn  = np.zeros((rows,cols),float)
    b = 0.99
    a = 0.01
    np.seterr(invalid='ignore')
    for j in range(0, cols ):
        aux_min =100000000000
        aux_max = 0
        for i in range(0, rows ):
            if j != cols -1 :
                if aux_min > x[i,j]:
                    aux_min = x[i,j]
                if aux_max < x[i,j]:
                    aux_max = x[i,j]
        #print("minimo",aux_min,"maximo", aux_max)
        for k in range(0, rows ):
            if j != cols:
                x_a = (x[k,j] - aux_min)
                x_b = (aux_max - aux_min)
                restab = (b-a)
                if np.isnan(((x_a / x_b) * restab ) + a):
                    xn[k,j] = 0
                else:
                    xn[k,j] = ((x_a / x_b) * restab ) + a

    return(xn)

def main():
    param = load_config_sv()
    start_time = time.time()
    #current_file = os.getcwd() + "\\fuentes\KDDTrain.txt"

    #print(X,y)
    #separar datos en X e y
    #order = sv.inf_gain(X,y)
    #print(order)
    #print(normalizar(X))
    print("*********************")
    x_train,y_train = sv.select_variables(param[2],param[3])


    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train.shape,y_train.shape)
    #x,y = inf_gain(X,y, par[2]) #parametro 2 es la proporcion de valores que se usará creo, en teoría features*par[2] = k
    #sacar v con svd, sacar con eso x nuevo 
    #guardar índice de características más importantes y filter_v
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':   
	 main()
