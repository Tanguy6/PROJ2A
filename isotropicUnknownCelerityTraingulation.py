
import scipy as sc
import statistics as stat


import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import random
import time

import math

class IMU:
    def __init__(self, x, y,t):
        self.x = x
        self.y = y
        self.t = t


Imu1 = IMU(0,0,0)
Imu2 = IMU(0,1,0)
Imu3 = IMU(0,2,0)
Imu4 = IMU(1,2,0)
Imu5 = IMU(2,2,0)
Imu6 = IMU(2,1,0)
Imu7 = IMU(2,0,0)
Imu8 = IMU(1,0,0)

n = 8

def tij(first,second): # Différence de temps d'arrivée
    return (first.t - second.t)*0.001 # Mettre des secondes au lieu des indices de tableau

def di(source,sensor): # Norme entre 2 points
    return math.sqrt(math.pow((sensor.x-source.x),2)+math.pow((sensor.y-source.y),2)) # Vérifier le calcul et les valeurs


# def toMinimize(x,y): 
#     return math.pow(2*(math.pow(x,2)+math.pow(y,2)-x*Imu2.x-x*Imu3.x-y*Imu2.y-y*Imu3.y)+(math.pow(Imu2.x,2) + math.pow(Imu3.x,2) + math.pow(Imu2.y,2) + math.pow(Imu3.y,2))-(math.pow((Imu2.t/Imu1.t),2) + math.pow((Imu3.t/Imu1.t),2))*(math.pow(x, 2) + math.pow(y, 2) - 2*(x*Imu1.x+y*Imu1.y) + math.pow(Imu1.x, 2) + math.pow(Imu1.y, 2)),2) 


def toMinimizeBis(coordonates): # Fonction à minimiser tirée de la revue de Kundu et al.
    Tab = [Imu1,Imu2,Imu3,Imu4,Imu5,Imu6,Imu7,Imu8]
    Imu9 = IMU(coordonates[0], coordonates[1], 0)
    toRet = 0
    for i in range(0, n-1):
        for j in range(i, n):
            for k in range(0, n-1):
                for l in range(k,n):
                    toRet += math.pow(  tij(Tab[i],Tab[j])*(di(Imu9, Tab[k]) - di(Imu9, Tab[l])) - tij(Tab[k],Tab[l])*(di(Imu9, Tab[i]) - di(Imu9, Tab[j])) ,2)
    return toRet

def findPeak(tab): # Implémentation naive pour trouver le TdA
    ss = len(tab)
    for i in range(ss):
        if abs(tab[i]) > 0.5: # À mettre à valeur positive
            return i
        
# Verification de la "bonne" détection en regardant que t=0 soit bien sur le premier IMU
        
def plotPoints(knownPoint,foundPoint,i): 
    plt.plot(foundPoint[1],foundPoint[0],'rx',label=f"found point {i}")
    plt.plot(knownPoint[0],knownPoint[1],'bx',label="known point")
    plt.ylim(0, 2)
    plt.xlim(0, 2)
    plt.legend(loc="upper left")
    plt.show()


def findPoint(CurrentImpactAccelero): # Réalise l'optimisation
    x0 = [0,0]
    res = sc.optimize.minimize(toMinimizeBis, x0, method='Nelder-Mead', tol=1e-6)
    return res.x
    

def initialize_IMU(CurrentImpactAccelero,CurrentIMULocalisations):
    # Initialisation du temps
    Imu1.t = findPeak(CurrentImpactAccelero[1])
    Imu2.t = findPeak(CurrentImpactAccelero[4])
    Imu3.t = findPeak(CurrentImpactAccelero[7])
    Imu4.t = findPeak(CurrentImpactAccelero[10])
    Imu5.t = findPeak(CurrentImpactAccelero[13])
    Imu6.t = findPeak(CurrentImpactAccelero[16])
    Imu7.t = findPeak(CurrentImpactAccelero[19])
    Imu8.t = findPeak(CurrentImpactAccelero[22])
    # Initialisation des positions en x
    Imu1.x = CurrentIMULocalisations[0][0]
    Imu2.x = CurrentIMULocalisations[1][0]
    Imu3.x = CurrentIMULocalisations[2][0]
    Imu4.x = CurrentIMULocalisations[3][0]
    Imu5.x = CurrentIMULocalisations[4][0]
    Imu6.x = CurrentIMULocalisations[5][0]
    Imu7.x = CurrentIMULocalisations[6][0]
    Imu8.x = CurrentIMULocalisations[7][0]
    # Initialisation des positions en y
    Imu1.y = CurrentIMULocalisations[0][1]
    Imu2.y = CurrentIMULocalisations[1][1]
    Imu3.y = CurrentIMULocalisations[2][1]
    Imu4.y = CurrentIMULocalisations[3][1]
    Imu5.y = CurrentIMULocalisations[4][1]
    Imu6.y = CurrentIMULocalisations[5][1]
    Imu7.y = CurrentIMULocalisations[6][1]
    Imu8.y = CurrentIMULocalisations[7][1]

def analysis(tabValue):
    print("Moyenne : " + str(stat.mean(tabValue)))
    print("Écart-type : " + str(stat.pstdev(tabValue)))
    print("Médiane : " + str(stat.median(tabValue)))
    plt.hist(tabValue,bins=100)
    plt.axvline(x = stat.mean(tabValue), color = 'r', label = 'Moyenne')
    plt.axvline(x = stat.pstdev(tabValue), color = 'g', label = 'Écart-type')
    plt.axvline(x = stat.median(tabValue), color = 'y', label = 'Médiane')
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper left')
    plt.show()

def main():
    ImpactAccelero = np.load("Data/impacteur_accelero.npy")
    ImpactLocalisation = np.load("Data/impacteur_localisation.npy")
    IMULocalisations = np.load("Data/impacteur_pos_accelero.npy") # Contient la position moyenne de chaque IMU en x,y,z pour chaque impact

    
    nb_impact = len(ImpactAccelero)
    
    allNormErrors = []
    for current_impact_index in range(100): # Pour le pic en valeur absolue, ça donne une valeur absurde pour 112
        if current_impact_index != 112:
            initialize_IMU(ImpactAccelero[current_impact_index],IMULocalisations[current_impact_index])
            foundPoint = findPoint(ImpactAccelero[current_impact_index])
            norm_Error = math.sqrt(math.pow((foundPoint[1]-ImpactLocalisation[current_impact_index][0][0]),2)+math.pow((foundPoint[0]-ImpactLocalisation[current_impact_index][1][0]),2))
            # print(norm_Error)
            allNormErrors.append(norm_Error)
        # plotPoints(ImpactLocalisation[current_impact_index],foundPoint,current_impact_index)
    analysis(allNormErrors)
    # for i in range(100):
    # ko = 112
    # plt.plot(ImpactAccelero[ko][1][10:20])
    # # plt.plot(ImpactAccelero[ko][4][10:60])
    # # plt.plot(ImpactAccelero[ko][7][10:60])
    # # plt.plot(ImpactAccelero[ko][10][10:60])
    # # plt.plot(ImpactAccelero[ko][13][10:60])
    # # plt.plot(ImpactAccelero[ko][16][10:60])
    # # plt.plot(ImpactAccelero[ko][19][10:60])
    # # plt.plot(ImpactAccelero[ko][22][10:60])
    # plt.show()
    # # print(IMULocalisations[111])
    # # print(IMULocalisations[112])
    # initialize_IMU(ImpactAccelero[ko],IMULocalisations[ko])
    # print(Imu1.t)
    # print(Imu2.t)
    # print(Imu3.t)
    # print(Imu4.t)
    # print(Imu5.t)
    # print(Imu6.t)
    # print(Imu7.t)
    # print(Imu8.t)
    # print(ImpactLocalisation[ko][0][0],ImpactLocalisation[ko][1][0])
    # # plotPoints(ImpactLocalisation[112],ImpactLocalisation[112],112)
  


if __name__ == "__main__":
    main()






