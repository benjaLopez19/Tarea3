# My Utility : Algorithm PSO 

import numpy    as np
import util_bp  as bp

#Swarm: Initializing
def iniSwarm(Nh,Np,d):
    '''
    Nh n de nodos
    Np n de particulas
    d n de características
    '''
    X = np.zeros((Np,(Nh*d+2*Nh))) #matriz de la forma 40 filas con CALETA DE columnas (n de nodos * caracteristicas + n de nodos*2)
    
    X = np.array(X)
    columna = X.shape[1]
    filas = X.shape[0]
    
    for i in range(0, filas):
        w1,w2 = bp.randW(Nh,d)
        #print("w1",w1.shape,"w2",w2.shape)
        w1 = np.reshape(w1,Nh*d)
        #print(w1.shape)
        w2 = np.reshape(w2,Nh*2)
        #print(w2.shape)
        X[i] = np.append(w1,w2)
        #print(X[i])
    #print(X.shape)
    #print("X en iniswarm", X)
    return(X)

    
# Fitness by use MSE
def Fitness_mse(x,y,X,act,Nh): 
    '''
    x: base de datos
    y: valores objetivo:
    X: Particulas
    act: funciond de activacion
    Nh: numero de nodos ocultos
    '''
    n = y.shape[1]
    d = x.shape[0] #cantidad de variables
    N = x.shape[1] #cantidad de muestras
    Np = X.shape[0] #cantidad de particulas
    MSE = [] #error cuadráticos medio
    #e = np.zeros((1,N))
    for i in range(Np): #Error por cada partícula
        p = X[i,:] #saca una partícula
        #print(p.shape)
        W1 = np.reshape(p[0:Nh*d],(Nh,d)) #se separan los pesos
        #print(W1.shape)
        W2 = np.reshape(p[Nh*d:],(2,Nh))
        #print(W2.shape)
        y_pred = bp.forward(x,W1,W2,act)[3]
        
        #print("YPRED",y_pred.shape)

        mseAux = ((np.linalg.norm(y_pred-y))**2) /N
        MSE.append(mseAux)
    
    return(MSE)
    
# Update:Particles based on Fitness-MSE
def upd_pFitness(P,costo,X):
    for i in range(X.shape[0]):
        if (costo[i] <P['Fit'][0][i]):
            #print("se mete al if")
            P['Fit'][0][i] = costo[i]
            P['Pos'][i][:]    = X[i,:]
    gfit = min(P['Fit'][0])
    posicion = np.argmin(P['Fit'][0])

    if (gfit < P['gFit']):
        P['gFit'] = gfit
        P['gBest'] = P['Pos'][posicion][:]
    #print("costo", costo)
    #print("gFit", P['gFit'])
    return(P)    

# Update: Swarm's velocity
def upd_veloc(P,V,X,iTerA, iTerT):
    a_min = 0.1
    a_max = 0.95
   
    c1 = 1
    c2 = 2
    a = a_max - (((a_max -a_min)/iTerT) * iTerA)
    print('a', a)
    #print("ramdon 1",r1,"ramdom2",r2)
    aux = np.copy(V)
    
    r1 = np.random.random()
    r2 = np.random.random()
    #print(P["Pos"][0],X[0])
    for i in range(0,V.shape[0]):
        #print('v[I]',V[i])
        #print("P['Pos'][i] - X[i]",P['Pos'][i] - X[i])
        #print("P['gBest']) - X[i]",(P['gBest']) - X[i])
        aux[i] = (a*V[i]) + ((c1*r1)*(P['Pos'][i] - X[i])) + ((c2*r2)*((P['gBest']) - X[i]))

    V = aux
    V = bound_vel(V)
    #print(V[0])
    print(X[0][0])
    X = X +V
    X = bound(X)
    #print('vel maxima',np.amax(V))
    return(V,X) 

def bound(a):

    #a = np.clip(a,-500,500,out=0)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] > 1:
                a[i][j] =0
            elif a[i][j] <-1:
                a[i][j] =0
            
    return a

def bound_vel(a):

    #a = np.clip(a,-500,500,out=0)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] > 20:
                a[i][j] =0
            elif a[i][j] <-20:
                a[i][j] =0
    '''
    for i in a:
        #print(i)
        for j in i:
            if j >= 1:
                
                #print(j)
                j =0
                #print(j)
            elif j <= -1:
                #print(i)
                j=0
                #print(j)
    '''
    return a
    
#-----------------------------------------------------------------------
