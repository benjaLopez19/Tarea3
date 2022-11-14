
import numpy   as np
import util_bp as bp

# Test: Load data 
def load_data():
    ...    
    return(x,y,w1,w2)
#Normalized data

#Measure
def metricas():
    ...
    return()



def main():			
	xv,yv,w1,w2 = load_data()	
	zv          = bp.forward()      		
	metricas(yv,zv) 	

# Beginning ...
if __name__ == '__main__':   
	 main()

