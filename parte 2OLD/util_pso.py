# My Utility : Algorithm PSO 

import numpy    as np
import util_bp  as bp

#Swarm: Initializing
def iniSwarm(Nh,Np,d):
    X = np.zeros((Np,(Nh*d)))
    
    X = np.array(X)
    columna = X.shape[1]
    filas = X.shape[0]
    
    for i in range(0, filas):
        a = bp.randW(Nh,d)
        X[i] = np.reshape(a,(1,(Nh * d)))
    #print(X.shape)
    return(X)
# Initialize random weights
# Fitness by use MSE
def Fitness_mse(x,y,X,C):
    Nh = int(X.shape[1]//x.shape[0])
    #print("nodos ocultos",Nh)
    d = x.shape[0]
    N = x.shape[1]
    Np = X.shape[0]
    w2 = []
    MSE = []
    e = np.zeros((1,N))
    for i in range(Np):
        p = X[i,:]
        MP = np.reshape(p,(Nh,d))
        #print(MP.T.shape)
        H = bp.act_function(MP,np.transpose(x))
        #print("MP SHAPE",MP.shape)
        w2.append(bp.ann_updW(H,y,C))
        Y = np.dot(np.transpose(w2[i]),H)
        z =np.transpose(y)
        #print("esto es Y",Y.shape, y.shape)
        suma = 0
        for j in range(Y.shape[1]):
            e[0,j] = np.square(z[0,j] - Y[0,j])
            suma = suma + e[0,j]
        #print(suma/Y.shape[1], np.square(np.subtract(y,z))/Y.shape[1] )
        MSE.append(suma/Y.shape[1])
    MSE= np.array(MSE)       
    return(w2,MSE)
    
# Update:Particles based on Fitness-MSE
def upd_pFitness(P,V,X,iTerA, iTerT):
    ...
    return()
# Update: Swarm's velocity
def upd_veloc(P,V,X,iTerA, iTerT):
    a_min = 0.1
    a_max = 0.95
   
    c1 = 1.05
    c2 = 2.95
    a = a_max - (((a_max -a_min)/iTerT) * iTerA)
    #print("ramdon 1",r1,"ramdom2",r2)
    aux = V
    aux_2 = X
    
    
    #for i in range(0,V.shape[0]):
    #    r1 = np.random.random()
    #    r2 = np.random.random()
       
     #   a = a_max - (((a_max -a_min)/V.shape[0]) * i)
        #print(P['Pos'][i], P['gBest'])
        #V[i+1] = (a * aux[i]) + (c1*r1) *(P['Pos'][i] - aux_2[i]) + (c2*r2)*(P['gBest'][i] - aux_2[i])
      #  for j in range(0,V.shape[1]-1 ):
            
            #print(V[i,j+1])
       #     V[i,j+1] = (a * aux[i,j]) + (c1*r1) *(P['Pos'][i][j] - X[i,j]) + (c2*r2)*(P['gBest'][j] - X[i,j])
            #if (V[i,j+1] > 0.5):
             #   V[i,j+1] = 0.5
        #    X[i,j+1] = aux_2[i,j] + V[i,j+1]
            #print(V[i,j+1],aux[i,j])
    #print(V.shape, X.shape, P['Pos'].shape, P['gBest'].shape)
    r1 = np.random.random()
    r2 = np.random.random()
    for i in range(0,V.shape[0]):
        
        aux[i] = (a * V[i]) + (c1*r1)*(P['Pos'][i] - X[i]) + (c2*r2)*((P['gBest']) - X[i])
  
    V = aux
    X = X +V
    return(V,X)  
  
    
#-----------------------------------------------------------------------