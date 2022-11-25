
import numpy   as np
import util_bp as bp
import trn


#Carga data de testing
#Crear archivo de matriz de confusion
    #cmatriz.csv (cuadrada)
# Crear archivo F-scores
    #fsocres.csv [Nx1] - ultima pos es el valor promedio de los fscores. previos.

# Test: Load data 
def load_data():
    x = np.loadtxt('xtst.csv', dtype= float, delimiter=",")
    y = np.loadtxt('ytst.csv', dtype= int, delimiter=",")
    y = np.transpose(y)

    with np.load('pesos_ann.npz') as archivo :
        w1 = archivo['x']
        w2 = archivo['y']

    return(x,y,w1,w2)

#Measure
def metricas(y_predict,z_test):
    z = np.array(z)
    metrics=[0,0,0]

    confusion = [[0,0],[0,0]]
    for i in range(0,y_predict.shape[0]):
        if int(y_predict[i,0]) == 1:
            if np.any(z_test[0,i] >= 0) :
                confusion[0][0] = confusion[0][0] + 1
            else:
                confusion[0][1] = confusion[0][1] + 1
        elif int(y_predict[i,0]) == 0:
            if np.any(z_test[0,i] < 0):
                confusion[1][0] = confusion[1][0] + 1
            else:
                confusion[1][1] = confusion[1][1] + 1

    np.savetxt('cmatrix_ann.csv', confusion, fmt='%i', header=' ',  delimiter=' ; ')

    matrix = confusion

    p1 = matrix[0][0]/(matrix[0][0] + matrix[0][1])
    r1 = matrix[0][0]/(matrix[0][0] + matrix[1][0])
    p2 = matrix[1][1]/(matrix[1][1] + matrix[1][0])
    r2 = matrix[1][1]/(matrix[1][1] + matrix[0][1])
    metrics[0] = float(2*((p1*r1)/(p1+r1)))
    metrics[1] = float(2*((p2*r2)/(p2+r2)))
    metrics[2] = (metrics[0]+metrics[1])/2
    np.savetxt('fscores_ann.csv', metrics, fmt='%.5f', header=' ',  delimiter=' ; ')
    return()



def main():			
    param = trn.load_config()
    xv, yv,w1,w2 = load_data()	
    zv          = bp.forward(xv,w1,w2,param[0])      		
    metricas(yv,zv)

    return() 	

# Beginning ...
if __name__ == '__main__':   
	 main()

