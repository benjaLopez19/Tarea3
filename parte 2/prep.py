# Pre-processing...

import pandas as pd
import numpy  as np
import random

#cargar archivos dtrn.csv y etrn.csv
#normalizar
#reordenar aleatoriamente las posiciones

#crear archivos de datos para training xtrn.csv, ytrn.csv
#los mismos pasos para testing creando archivos xtst.csv, ytst.csv


# Load data 
def load_data(type):
    '''Carga la informacion de los archivos
    Crea los archivos de datos 
    
    Parameter type: 0 para training o 1 para test'''

    if type == 0:
        dtrn = np.loadtxt('dtrn.csv', dtype= float, delimiter=",")
        etrn = np.loadtxt('etrn.csv', dtype= int, delimiter=",")

        xtrn = norma_data(dtrn)
        #index = random.shuffle(etrn)
        #xtrn = 

        ytrn = label_binary(etrn)
        print("hola")
        np.savetxt('xtrn.csv', xtrn, fmt='%.3f',  delimiter=' , ')
        np.savetxt('ytrn.csv', ytrn, fmt='%d',  delimiter=' , ')

    if type == 1:
        dtst = np.loadtxt('dtst.csv', dtype= float, delimiter=",")
        etst = np.loadtxt('etst.csv', dtype= int, delimiter=",")

        xtst = norma_data(dtst)
        random.shuffle(xtst)

        ytst = label_binary(etst)

        np.savetxt('xtst.csv', xtst, fmt='%d', header=' ',  delimiter=' , ')
        np.savetxt('ytst.csv', ytst, fmt='%d', header=' ',  delimiter=' , ')

    return()

#Normazationg of the features 
def norma_data(x):
    '''Normaliza el array de informacion'''
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


# Create binary label
#crear etiquetas binarias, vector con clases 1,2 y 3 pasarlo a una matriz Nx3
# 1 = 1,0,0; 2 = 0,1,0; 3 = 0,0,1 y- 1,2,1,3,2 - [1,0,0],[0,1,0]...
def label_binary(data):
    '''Transforma las etiquetas en binario'''
    ydata = []
    for x in data:
        match x:
            case 1:
                ydata.append([1,0])
            case 2:
                ydata.append([0,1])
            #case 3:
                #ydata.append([0,0,1])
        continue

    return(ydata)

def main():
    load_data(0)
    return()
    
if __name__ == '__main__':   
	 main()
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

