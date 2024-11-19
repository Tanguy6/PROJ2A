import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import math 


# Contient les implémentations des différentes méthodes pour trouver le début d'un signal

from classDefinition import IMU

def crossCorrelation(signal1,signal2):
    result = scp.signal.correlate(signal1, signal2,mode='full', method='auto') # Fait glisser le premier signal sur le deuxième en partant de la droite du tableau
    # print(result)
    # print(max(result))
    return np.argmax(result) - len(signal1) # Le deltaT en indice

def testCrossCorrelation():
    x = np.arange(0,2*np.pi,0.00001*np.pi) 
    xx = np.arange(np.pi/2,2*np.pi+np.pi/2,0.00001*np.pi)
    # print(x)
    y = np.sin(x)
    yy = np.cos(xx)
    print(crossCorrelation(y, yy)) # Doit sortir environ +/- 100000, i.e. pour avoir une différence de pi vu le pas de temps
    # plt.plot(x,y,x,yy)
    # plt.show()
    
def hilbertEnveloppe(signal1):
    hilbertTransform = scp.signal.hilbert(signal1)
    enveloppe = np.abs(hilbertTransform)
    return findPeak(enveloppe)
    # Et il faut mettre un seuil sur cette enveloppe. Là encore, à voir ...

def testhilbertEnveloppe():
    duration = 1.0
    fs = 400.0
    samples = int(fs*duration)
    t = np.arange(samples) / fs
    signal = scp.signal.chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
    hilbertTransform = scp.signal.hilbert(signal)
    enveloppe = np.abs(hilbertTransform)
    plt.plot(t,signal,t,enveloppe)
    
  
def findPeak(tab,seuilValue): # Implémentation naive pour trouver le TdA
    ss = len(tab)
    for i in range(ss):
        if abs(tab[i]) > seuilValue: # À mettre à valeur positive
            return i
      
def tij(first,second,seuilValue): # Différence de temps d'arrivée
    # return (first.t - second.t)*0.001 # Mettre des secondes au lieu des indices de tableau    
    return (findPeak(first.t,seuilValue) - findPeak(second.t,seuilValue))

def di(source,sensor): # Norme entre 2 points
    return math.sqrt(math.pow((sensor.x-source.x),2)+math.pow((sensor.y-source.y),2)) # Vérifier le calcul et les valeurs



def main():
    testCrossCorrelation()
    print("End Main")
    
    
if __name__ == "__main__":
    main()