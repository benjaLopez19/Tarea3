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
def load_data():
    '''Carga la informacion de los archivos
    Crea los archivos de datos 
    '''

    index = np.loadtxt('index_var.csv', dtype= int, delimiter=";")
    v = np.loadtxt('filter.csv', dtype= int, delimiter=";")

    ################Training##############################
    dtrn = np.loadtxt('dtrn.csv', dtype= float, delimiter=",")
    etrn = np.loadtxt('etrn.csv', dtype= int, delimiter=",")
    #A = np.loadtxt('fuentes\KDDTrain.txt', dtype= int, delimiter=",")
    #print("shaaaape",dtrn.shape)
    A = np.vstack([dtrn, etrn])
    A = np.transpose(A)
    np.random.shuffle(A)
    A= np.transpose(A)
    dtrn = A[0:40,:]
    etrn = A[42,:]
    #print(etrn)

    dtrn = dtrn[index,:]
    #filtrar con v (vtraspuesta*x)
    xtrn = np.dot(np.transpose(v),dtrn)  

    xtrn = norma_data(xtrn)
    #print(xtrn.shape)
    ytrn = label_binary(etrn)
    #print(ytrn.shape)
    np.savetxt('xtrn.csv', xtrn, fmt='%.3f',  delimiter=' , ')
    np.savetxt('ytrn.csv', ytrn, fmt='%d',  delimiter=' , ')

    ################Testing##############################
    dtst = np.loadtxt('dtst.csv', dtype= float, delimiter=",")
    etst = np.loadtxt('etst.csv', dtype= int, delimiter=",")

    A = np.vstack([dtst, etst])
    A = np.transpose(A)
    np.random.shuffle(A)
    A= np.transpose(A)
        
    dtst = A[0:40,:]
    etst = A[42,:]

    dtst = dtst[index,:]
    #filtrar con v (vtraspuesta*x)
    dtst = np.dot(np.transpose(v),dtst)  

    xtst = norma_data(dtst)

    ytst = label_binary(etst)

    np.savetxt('xtst.csv', xtst, fmt='%.3f',  delimiter=' , ')
    np.savetxt('ytst.csv', ytst, fmt='%d',  delimiter=' , ')

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
    
    ydata = np.array(ydata)

    return(ydata)

def main():
    load_data()
    return()
    
if __name__ == '__main__':   
	 main()
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

