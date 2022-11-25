# My Utility : Back-Propagation

import pandas as pd
import numpy  as np   
 

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
    par.append(np.float32(paramAnn[1])) # Tasa de Aprendizaje 

    return(par)


# Randomized Weight 
def randW(Nh,m):
    '''
    Nh cantidad nodos ocultos
    m numero de variables
    '''
    r = np.sqrt(6/(Nh+m))
    W1 = np.random.rand(Nh, m) * 2* r- r
    r = np.sqrt(6/(Nh+2))
    W2 = np.random.rand(2, Nh) * 2* r- r
    return(W1,W2)

# Feed-forward of the ANN
def forward(x, W1, W2, n_param):
    '''
    x base de datos
    W1 pesos capa oculta
    W2 pesos capa salida
    n_param numero de funcion de activacion
    '''
    #W1, W2 = 0 #Se separan los pesos
    
    Z1 = W1.dot(x)
    H = act_function(Z1,n_param)
    Z2 = W2.dot(H)
    y_pred = act_function(Z2,5)
    #print(y_pred)
    y_pred = np.array(y_pred)
    print(y_pred.shape)
    return([Z1,H,Z2,y_pred])

#Activation function
#como programarla
def act_function(z,n_param):
    z = np.clip(z,-500,500)

    match n_param:
        case 1: #ReLu
            return np.maximum(0,z)

        case 2: #L-ReLu
            if z<0:
                return z*0.01
            else:
                return z
        case 3: #ELU
            alpha = 1.6732
            return np.where(z > 0, z, alpha * (np.exp(z) - 1))

        case 4: #SELU
            alpha = 1.6732
            Lambda = 1.0507
            if z<=0:
                return Lambda*(alpha*(np.exp(z)-1))
            else:
                return Lambda*z

        case 5: #Sigmodidal
            return 1/(1+np.exp(-z))

# Derivative of  Activation function    
def derivate_act(z,n_param):
    match n_param:
        case 1: #ReLu
            return z > 0
        
        case 2: #L-ReLu
            out = np.ones_like(z)
            out[z < 0] *= 0.01

            return out

        case 3: #ELU
            return np.where(z > 0, np.ones_like(z), 0.01 * np.exp(z))

        case 4: #SELU
            alpha = 1.6732
            Lambda = 1.0507
            if z<=0:
                return Lambda*(alpha*(np.exp(z)-1))
            else:
                return Lambda*z
        
        case 5: #Sigmodidal
            #return z*(1-z)
            return  1/(1+np.exp(-z)) * (1-1/(1+np.exp(-z))) #sigmoid(z)*(1-sigmoid(z))
    return()
# STEP 2: Feed-Backward: 
def ann_gradW(m,salida,n_param,y,W1,W2,X):#    
    '''
    m numero de variables
    salida parametros entregados por forward: [Z1,H,Z2,y_pred]
    n_param funcion activacion capa oculta
    y valores reales
    W1 pesos capa oculta
    W2 pesos capa salida
    X base de datos
    '''

    error_capa_2 = salida[3] - y
    act_derivada = np.vectorize(derivate_act)

    delta_capa_2 = np.multiply(error_capa_2,act_derivada(salida[3],5)) # error por f'(z) producto hadamar
    gradiente_salida = delta_capa_2 @ salida[1].T #delta(o)*H.t

    error_capa_1 = W2.T @ delta_capa_2 # V.T * delta(o
    delta_capa_1 = np.multiply(error_capa_1, act_derivada(salida[1],n_param)) #error*H'(z) producto hadamar también
    gradiente_oculta = delta_capa_1 @ X.T #delta(h)*X.T


    return gradiente_oculta, gradiente_salida
    
  
# Update Ws
def ann_updW(alpha, W1, W2, dW1, dW2):

    W1 = W1 - alpha * dW1    
    W2 = W2 - alpha * dW2  
 
    return W1, W2
#-----------------------------------------------------------------------

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def main():
    param = load_config()
    x = np.loadtxt('xtrn.csv', dtype= float, delimiter=",")
    #x = np.transpose(x)
    y = np.loadtxt('ytrn.csv', dtype= int, delimiter=",")
    y = np.transpose(y)
    m = x.shape[0]
    print(x.shape,y.shape)
    print(param)
 
    W1, W2 = randW(param[1], m)
    print(W1.shape,W2.shape)
    
    salida = forward(x, W1, W2, param[0])
    salida = np.array(salida)
    print(salida[3].shape)
    print(salida[3])

    for i in range(param[4]):
        salida = forward(x, W1, W2, param[0])
        #print("salida",salida[3])
        dW1, dW2 = ann_gradW(m, salida, param[0],y,W1,W2,x) 
        W1, W2 = ann_updW(param[5],W1,W2,dW1, dW2)
        if (i % 10 == 0):
            print("Iteration: ", i)
            predictions = get_predictions(salida[3])
            print(salida[3].shape)
            print(salida[3][0:3])
            print("ACC" ,get_accuracy(predictions, y))
    #print(W1,W2)
    
    salida = forward(x, W1, W2, param[0])
    
    print(salida[3])
    #print(y)
    return ()

if __name__ == '__main__':   
	main()