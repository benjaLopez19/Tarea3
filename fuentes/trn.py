import numpy as np
import myutility as mt
import sv

#main
#x,y = selectvariables
#guardar x como dtrn.csv
#guardar y como etrn.csv

def main():
    x,y = sv.select_variables()
    np.savetxt('dtrn.csv', x, fmt='%d', header=' ',  delimiter=' , ')
    np.savetxt('etrn.csv', y, fmt='%d', header=' ',  delimiter=' , ') 


if __name__ == '__main__':   
	 main()