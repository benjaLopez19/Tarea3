
import numpy   as np
import util_bp as bp

#Carga data de testing
#Crear archivo de matriz de confusion
    #cmatriz.csv (cuadrada)
# Crear archivo F-scores
    #fsocres.csv [Nx1] - ultima pos es el valor promedio de los fscores. previos.

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
	zv          = bp.forward(xv,[w1,w2])      		
	metricas(yv,zv) 	

# Beginning ...
if __name__ == '__main__':   
	 main()

