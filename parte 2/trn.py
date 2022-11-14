# Training: IDS ANN-PSO-BP

import pandas   as pd
import numpy    as np
import util_pso as pso
import util_bp  as bp

# Load Parameters
def load_config():
    ...
    return(param)
# Load data of training
def load_data():
	...
	return(x,y)
#save weights in numpy format
def save_w():
    ...
    return

# Training: ANN-BP
def ann_bp(w1,w2,x,y,..):
    ...    
    return(w1,w2)

# Training : ANN-PSO
def ann_pso(x,y,...):
    ...    
    return(w1,w2)
# Training:ANN-PSO and  ANN-BP
def train_ann(x,y,param):
    w1,w2 = ann_pso(x,y,...)
    w1,w2 = ann_bp(w1,w2,x,y,...)    
    return(w1,w2) 
   
# Beginning ...
def main():
    param   = load_config()            
    xe,ye   = load_data()   
    w1,w2   = ann_train(xe,ye,param)             
    save_w(w1,w2)
       
if __name__ == '__main__':   
	 main()

