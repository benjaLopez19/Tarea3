# My Utility : Back-Propagation

import pandas as pd
import numpy  as np   

# Randomized Weight 
def randW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Feed-forward of the ANN
def forward(x,W):        
    ...    
    return()

#Activation function
def act_function(z):
    ...
    return()
# Derivative of  Activation function    
def derivate_act():
    ...
    return()
# STEP 2: Feed-Backward: 
def ann_gradW(a,x,w):    
    ...    
    return()    
# Update Ws
def ann_updW():
    ...
    return()
#-----------------------------------------------------------------------