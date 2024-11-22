import scipy as sc
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import random
import time
import math

from classDefinition import IMU








Imu1 = IMU(0,0,0)
Imu2 = IMU(0,1,0)
Imu3 = IMU(0,2,0)
Imu4 = IMU(1,2,0)
Imu5 = IMU(2,2,0)
Imu6 = IMU(2,1,0)
Imu7 = IMU(2,0,0)
Imu8 = IMU(1,0,0)

n = 8

Tab = [Imu1,Imu2,Imu3,Imu4,Imu5,Imu6,Imu7,Imu8]


def findPeak(tab,seuilValue): # Implémentation naive pour trouver le TdA
    ss = len(tab)
    for i in range(ss):
        if abs(tab[i]) > seuilValue: # À mettre à valeur positive
            return i

def tij(first,second,seuilValue): # Différence de temps d'arrivée
    # return (first.t - second.t)*0.001 # Mettre des secondes au lieu des indices de tableau    
    # print(first.t)
    return (findPeak(first.t,seuilValue) - findPeak(second.t,seuilValue))


def di(source,sensor): # Norme entre 2 points
    return math.sqrt(math.pow((sensor.x-source.x),2)+math.pow((sensor.y-source.y),2)) # Vérifier le calcul et les valeurs


def trilaterationMethod(coordonates): # Fonction à minimiser tirée de la revue de Kundu et al.
    n = len(Tab)
    print(Tab[1].t)
    Imu9 = IMU(coordonates[0], coordonates[1], 0)
    toRet = 0
    for i in range(0, n-1):
        for j in range(i, n):
            for k in range(0, n-1):
                for l in range(k,n):
                    print(i)
                    toRet += math.pow(tij(Tab[i],Tab[j],valeurSeuil)*(di(Imu9, Tab[k]) - di(Imu9, Tab[l])) - tij(Tab[k],Tab[l],valeurSeuil)*(di(Imu9, Tab[i]) - di(Imu9, Tab[j])) ,2)
    return toRet


from classDefinition import Prediction
from classDefinition import dataVisualizer
from classDefinition import JSON_FILE
from classDefinition import Finder

# def findPeak(tab): # Implémentation naive pour trouver le TdA
#     ss = len(tab)
#     for i in range(ss):
#         if abs(tab[i]) > 0.5: # À mettre à valeur positive
#             return i
        
# Verification de la "bonne" détection en regardant que t=0 soit bien sur le premier IMU
        
def plotPoints(knownPoint,foundPoint,i): 
    plt.plot(foundPoint[1],foundPoint[0],'rx',label=f"found point {i}")
    plt.plot(knownPoint[1],knownPoint[0],'bx',label="known point")
    plt.ylim(-0.1,1.9)
    plt.xlim(-0.1,1.9)
    plt.legend(loc="upper left")
    plt.show()
    
def initialize_IMU(CurrentImpactAccelero,CurrentIMULocalisations,traitementAccelerometreParam):  
    match traitementAccelerometreParam:
        case "AxeZ":
            Imu1.t = CurrentImpactAccelero[1]
            Imu2.t = CurrentImpactAccelero[4]
            Imu3.t = CurrentImpactAccelero[7]
            Imu4.t = CurrentImpactAccelero[10]
            Imu5.t = CurrentImpactAccelero[13]
            Imu6.t = CurrentImpactAccelero[16]
            Imu7.t = CurrentImpactAccelero[19]
            Imu8.t = CurrentImpactAccelero[22]
        case "Norme":
            # On récupère la norme du vecteur accélération
            m11 = np.transpose(np.linalg.norm(CurrentImpactAccelero[0:3,:],axis=0))
            m12 = np.transpose(np.linalg.norm(CurrentImpactAccelero[3:6,:],axis=0))
            m13 = np.transpose(np.linalg.norm(CurrentImpactAccelero[6:9,:],axis=0))
            m21 = np.transpose(np.linalg.norm(CurrentImpactAccelero[9:12,:],axis=0))
            m23 = np.transpose(np.linalg.norm(CurrentImpactAccelero[12:15,:],axis=0))
            m31 = np.transpose(np.linalg.norm(CurrentImpactAccelero[15:18,:],axis=0))
            m32 = np.transpose(np.linalg.norm(CurrentImpactAccelero[18:21,:],axis=0))
            m33 = np.transpose(np.linalg.norm(CurrentImpactAccelero[21:24,:],axis=0))
            
            M11 = np.zeros(np.shape(m11))
            M12 = np.zeros(np.shape(m12))
            M13 = np.zeros(np.shape(m13))
            M21 = np.zeros(np.shape(m21))
            M23 = np.zeros(np.shape(m23))
            M31 = np.zeros(np.shape(m31))
            M32 = np.zeros(np.shape(m32))
            M33 = np.zeros(np.shape(m33))
            
            # On enlève l'offset sur les mesures (Par exemple la gravité)
            for i in range (len(m11)):
                M11[i] = m11[i]-m11[0]
                M12[i] = m12[i]-m12[0]
                M13[i] = m13[i]-m13[0]
                M21[i] = m21[i]-m21[0]
                M23[i] = m23[i]-m23[0]
                M31[i] = m31[i]-m31[0]
                M32[i] = m32[i]-m32[0]
                M33[i] = m33[i]-m33[0]
                
            Imu1.t = M11
            Imu2.t = M12
            Imu3.t = M13
            Imu4.t = M21
            Imu5.t = M23
            Imu6.t = M31
            Imu7.t = M32
            Imu8.t = M33  
        case _:
            print("Cette méthode pour récupérer l'accélération n'est pas valable.")
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



# Voir les options disponibles dans la classe "Prediction" du fichier classDefinition
typeLocalisation = "Trilateration"
typeTdA = "SeuilNaif"
typeOptimisation = "Default" 
valeurSeuil = 7
traitementAccelerometre = "Norme"
dataSet = "ImpactStage"

# def findPoint(CurrentImpactAccelero): # Réalise l'optimisation
#     x0 = [0,0]
#     # Bounds et point de départ ( qui ne change pas forcément grand chose ) 
#     res = sc.optimize.minimize(trilaterationMethod, x0, method=typeOptimisation, tol=1e-6)
#     return res.x

# def findPeak(tab): # Implémentation naive pour trouver le TdA
#     ss = len(tab)
#     for i in range(ss):
#         if abs(tab[i]) > valeurSeuil: # À mettre à valeur positive
#             return i



   

def main():

    finder = Finder(typeLocalisation,typeTdA,typeOptimisation,valeurSeuil,traitementAccelerometre,dataSet)
    finder.chargerDataSet()
    # (ImpactAccelero, ImpactLocalisation, IMULocalisations) = chargerDataSet(dataSet)
    # nb_impact = len(ImpactAccelero)
    
    allNormErrors = []
    
    
    for current_impact_index in range(0,10): # Pour le pic en valeur absolue, ça donne une valeur absurde pour 112
        if current_impact_index != 112:
            finder.initialize_IMU(current_impact_index)
             # initialize_IMU(ImpactAccelero[current_impact_index],IMULocalisations[current_impact_index],traitementAccelerometre)
             # Tab = [Imu1,Imu2,Imu3,Imu4,Imu5,Imu6,Imu7,Imu8]
             # print(Tab[0].t)
            foundPoint = finder.getPredictedPoint()
            norm_Error = math.sqrt(math.pow((foundPoint[0]-finder.getRealPoint(current_impact_index)[0][0]),2)+math.pow((foundPoint[1]-finder.getRealPoint(current_impact_index)[1][0]),2))
            allNormErrors.append(norm_Error)
            plotPoints(finder.getRealPoint(current_impact_index),foundPoint,current_impact_index)
    
    # analysis(allNormErrors)
    # pred = Prediction("Trilateration", "CrossCorrelation")
    # pred.addData(allNormErrors)
    # pred.saveToJson()
    # test = dataVisualizer(JSON_FILE)
    # test.showData()
    # test.compareData(["Trilateration","Trilateration"], ["SeuilNaif","CrossCorrelation"])
  


if __name__ == "__main__":
    main()






