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
    x = load_data(r'D:\Cosas\Desktop\Universidad\Decimo semestre\Sistemas distribuidos\Tarea\Tarea3\fuentes\KDDTrain.txt')
    print(x)
    print("*********************")
    #print(y)
       
if __name__ == '__main__':   
	 main()
