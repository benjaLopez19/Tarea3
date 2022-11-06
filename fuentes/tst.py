import numpy as np
import myutility as mt
import sv

#main
#sacar archivo de indices
#sacar archivo de matriz v
#cargar datos de testeo
#filtrar/caracteristicas filas por indice
#filtrar con v (vtraspuesta*x)
#crear x como dtst.csv
#guardar y como etst.csv

def main():
    x,y = sv.select_variables()
    np.savetxt('dtrn.csv', x, fmt='%d', header=' ',  delimiter=' , ')
    np.savetxt('etrn.csv', y, fmt='%d', header=' ',  delimiter=' , ') 


if __name__ == '__main__':   
	 main()