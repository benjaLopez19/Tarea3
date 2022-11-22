# Training: IDS ANN-PSO-BP

import pandas   as pd
import numpy    as np
import util_pso as pso
import util_bp  as bp

#Entrenar con PSO
    #Crear archivo de costo del MSE
        #costo_pso.csv [Nx1]
#Entrenar con backpropagation
    #crear archivo de costo del MSE
        #costo_gd.csv [Nx1]
#Crear archivo de pesos entrenados
    #pesos.npz

# Load Parameters
def load_config():
    '''
    pso

0 : Número Activación Oculta : 2
1 : Número Nodos Ocultos : 20
2 : Número de Partículas : 25
3 : Número Iteraciones : 500

bp

Línea 1 : Número Iteraciones : 2000
Línea 2 : Tasa de Aprendizaje : 0.1
    '''

    paramAnn = np.genfromtxt("parte 2\cnf_ann_bp.csv",delimiter=',',dtype=None)
    paramPso = np.genfromtxt("parte 2\cnf_ann_pso.csv",delimiter=',',dtype=None)   
    par=[]    
    par.append(np.int16(paramPso[0])) # Número Activación Oculta
    par.append(np.int16(paramPso[1])) # Número Nodos Ocultos : 20
    par.append(np.int16(paramPso[2])) # Número de Partículas
    par.append(np.int16(paramPso[3])) # Número Iteraciones

    par.append(np.int16(paramAnn[0])) # Número Iteraciones     
    par.append(np.int16(paramAnn[1])) # Tasa de Aprendizaje 

    return(par)
    
# Load data of training
def load_data():
    x = np.loadtxt('xtrn.csv', dtype= float, delimiter=",")
    #x = np.transpose(x)
    y = np.loadtxt('ytrn.csv', dtype= int, delimiter=",")
    y = np.transpose(y)

    return(x,y)

#save weights in numpy format
def save_w():
    #...
    return

# Training: ANN-BP
def ann_bp(w1,w2,x,y):
    #PREGUNTAR BIEN LA ESTRUCTURA DEL BACKWARD
    #...    
    return(w1,w2)

# Training : ANN-PSO
def ann_pso(x,y,param):    
    X = pso.iniSwarm(param[1],param[2],x.shape[0])
    P = {}
    P['Pos']   = np.zeros(X.shape)               #Best particle position
    print(P['Pos'].shape)
    P['Fit']   = np.ones((1,X.shape[0]))*np.inf  #Best particle fitness
    P['gBest'] = np.zeros((1,X.shape[1]*2))        #Best global solution
    #print(P['gBest'].shape)
    P['gFit']  = np.inf                          #Best global Fitness
    V          = np.zeros(X.shape)               #Velicity  Initial        
    Cost  = []
    
    for iTer in range(param[3]):
        costo = pso.Fitness_mse(x,y,X, param[0], param[1]) #base de datos, objetivo real, particulas, activación
        P = pso.upd_pFitness(P,costo,X)
        #print(P["Pos"][0])
        #X = ut.upd_swarm(param[0], iTer, X, P['Pos'], P['gBest'])
        V,X = pso.upd_veloc(P,V,X,iTer,param[0])

        #X = X + V
        Cost.append(P['gFit'])
        print("BEST FITNESS",P['gFit'])
        if ((iTer % 10)== 0):
            print('Iter={} Cost= {:.5f}'.format(iTer,Cost[-1])) 

    w= P['gBest']
    #print(w.shape)
    #print(w[x.shape[0]:])
    #print(param[1],"*",x.shape[0]) 

    #print(param[1]*x.shape[0])
    W1 = np.reshape(w[0:param[1]*x.shape[0]],(param[1],x.shape[0])) #se separan los pesos
    W2 = np.reshape(w[param[1]*x.shape[0]:],(2,param[1]))
    #
    return(W1,W2)
    #PREGUNTAR QUÉ WEA ES EL PESO 2
    #PREGUNTAR FORMATO DE RED Y PESOS

# Training:ANN-PSO and  ANN-BP
def train_ann(x,y,param):
    w1,w2 = ann_pso(x,y,param)
    #w1,w2 = ann_bp(w1,w2,x,y,...)    
    return(w1,w2) 
   
# Beginning ...
def main():
    param   = load_config()           
    xe,ye   = load_data()   
    w1,w2   = train_ann(xe,ye,param)             
    #save_w(w1,w2)
       
if __name__ == '__main__':   
	 main()

