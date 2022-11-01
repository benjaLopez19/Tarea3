import numpy as np
import pandas as pd

#LECTURA ARCHIVO

def load_config_sv():      
    param = np.genfromtxt("configs/cnf_sv.csv",delimiter=',',dtype=None)    
    par=[]    
    par.append(np.int16(param[0])) # N. muestrar train 
    par.append(np.int16(param[1])) # N muestras test
    par.append(np.int16(param[2])) # Valor de relevancia
    par.append(np.float(param[3])) # N vectores singulares  
    par.append(np.int16(param[4])) # Clase normal     
    par.append(np.int16(param[5])) # Clase DOS
    par.append(np.int16(param[5])) # Clase Probe
    return(par)


def load_data(fname):
    """
    Carga datos desde el txt
    Para cada columna a excepción de la 42 (tipo de ataque) crear un diccionario con todos los tipos de valores individuales que tenga (particiones xd) y asignar un valor númerico a estos (los que no sean numéricos de por si), reemplazar en la matriz por estos valores
    Para la columna objetivo cambiar los valores de la siguiente manera
        Clase #1: Valores
		'normal'
		Clase #2: Valores
		'neptune', 'teardrop', 'smurf', 'pod', 'back',
		'land', 'apache2', 'processtable', 'mailbomb',
		'udpstorm'
		Clase #3: Valores
		'ipsweep', 'portsweep', 'nmap',
        'satan', 'saint', 'mscan'
    Normalizar las variables (puede ser entre 0.01 y 0.99) de las columnas 1 a 41


    """
    x  = np.loadtxt(fname, dtype= str, delimiter=",")
    
    '''
    x  = np.array(x)
    rows = x.shape[0]
    cols = x.shape[1]
    y  = np.zeros((rows, cols), float)
    aux= np.zeros((rows, 1), float)
    cont =0
    xn = np.zeros((rows,cols-1))
    for i in range (rows):
        for j in range (cols-1):
            xn[i,j] = x[i,j]
    #print("shape xn ",xn.shape)
            
    for j  in range(cols -1, cols):
        for i in range(0, rows):
            cont = cont+1
            y[i] = x[i,j]
            aux[i,0] = x[i,j]
    #xn = norm_data(xn)
    #print(" data",xn)
    
    return(np.transpose(xn),aux)
    '''
    return x

def main():
    datos = load_data(r'D:\Cosas\Desktop\Universidad\Decimo semestre\Sistemas distribuidos\Tarea\Tarea3\fuentes\KDDTrain.txt')
    #separar datos en X e y
    print("*********************")
    #x,y = inf_gain(X,y, par[2]) #parametro 2 es la proporcion de valores que se usará creo, en teoría features*par[2] = k
    #sacar v con svd, sacar con eso x nuevo 
    #guardar índice de características más importantes y filter_v    
if __name__ == '__main__':   
	 main()
