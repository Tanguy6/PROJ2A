import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

# Contient les implémentations des différentes méthodes pour trouver le début d'un signal


def crossCorrelation(signal1,signal2):
    result = scp.signal.correlate(signal1, signal2,mode='full', method='auto') # Fait glisser le premier signal sur le deuxième en partant de la droite du tableau
    print(result)
    print(max(result))
    return np.argmax(result) - len(signal1) # Le deltaT en indice

def testCrossCorrelation():
    x = np.arange(0,2*np.pi,0.00001*np.pi) 
    xx = np.arange(np.pi/2,2*np.pi+np.pi/2,0.00001*np.pi)
    print(x)
    y = np.sin(x)
    yy = np.cos(xx)
    print(crossCorrelation(y, yy)) # Doit sortir environ +/- 100000, i.e. pour avoir une différence de pi vu le pas de temps
    # plt.plot(x,y,x,yy)
    # plt.show()
    
    
def main():
    # testCrossCorrelation() # Affiche 13, avec les arrondis j'imagine
    print("End Main")
    
    
if __name__ == "__main__":
    main()