
# Tout est rassemble dans un seul code pour accelerer l'execution, par rapport
# a l'optimisation qui fait beaucoup d'appel fonction.


#######################################################
#                    Import
#######################################################

import scipy as sc
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import random
import time
import math
import json

#######################################################
#                    Constantes
#######################################################

JSON_FILE = "data.json"

# TYPE_LOCALISATION : "Trilateration" , "NeuralNetwork"
TYPE_LOCALISATION = "Trilateration"
# TYPE_TDA : "SeuilNaif" , "CrossCorrelation" , "SeuilEnveloppe" , "TransforméeOndelette"
TYPE_TDA = "SeuilNaif"
# TYPE_OPTIMISATION : "Default", "Nelder-Mead" , "Powell" , "CG" , "BFGS" , "Newton-CG" , "L-BFGS-B" , "TNC" , "COBYLA" , "SLSQP" , "trust-constr" , "dogleg" , "trust-ncg" , "trust-exact" , "trust-krylov"
TYPE_OPTIMISATION = "Default" 
# VALEUR_SEUIL : x
VALEUR_SEUIL = 7
# TRAITEMENT_ACCELEROMETRE : "AxeZ" , "Norme" 
TRAITEMENT_ACCELEROMETRE = "Norme"
# DATA_SET : "SautStage" , "ImpactStage", "ToutStage" , "SautMiniProj" , "ImpactMiniProj" , "ToutMiniProj" , "Tout"
DATA_SET = "ImpactStage"


#######################################################
#                    Classes
#######################################################

class IMU:
    def __init__(self, x, y,t):
        self.x = x
        self.y = y
        self.t = t

### On les affecte ici pour simplifier l'écriture

Imu1 = IMU(0,0,0)
Imu2 = IMU(0,1,0)
Imu3 = IMU(0,2,0)
Imu4 = IMU(1,2,0)
Imu5 = IMU(2,2,0)
Imu6 = IMU(2,1,0)
Imu7 = IMU(2,0,0)
Imu8 = IMU(1,0,0)

Tab = [Imu1,Imu2,Imu3,Imu4,Imu5,Imu6,Imu7,Imu8]

n = len(Tab)

class Prediction:
    
    
    # On donne ici ce qui définit une prédiction, c'est-à-dire l'ensemble
    # des paramètres sur lesquels on peut jouer.
    # typeLocalisation : "Trilateration" , "NeuralNetwork"
    # typeTdA : "SeuilNaif" , "CrossCorrelation" , "SeuilEnveloppe" , "TransforméeOndelette"
    # typeOptimisation : "Nelder-Mead" , "" 
    # valeurSeuil : int
    # traitementAccelerometre : "AxeZ" , "Norme" 
    # dataSet : "SautStage" , "ImpactStage", "ToutStage" , "SautMiniProj" , "ImpactMiniProj" , "ToutMiniProj" , "Tout"
    
    
    # Statique
    # Différence statistique : ANOVA
    
    
    def __init__(self, paramTypeLocalisation, paramTypeTdA):
        self.typeLocalisation = paramTypeLocalisation
        self.typeTdA = paramTypeTdA
        
    def addData(self,paramData):
        self.data = paramData
        
    def saveToJson(self):
        tempSuperDict = {}
        with open(JSON_FILE) as f:
            tempSuperDict = json.load(f)
        tempDict = {}
        tempDict["typeLocalisation"] = self.typeLocalisation
        tempDict["typeTdA"] = self.typeTdA
        tempDict["data"] = self.data
        tempSuperDict["Predictions" + str(len(tempSuperDict)+1)]= tempDict
        with open(JSON_FILE, 'w') as f:
            json.dump(tempSuperDict, f)  
            
class dataVisualizer:
    
    def __init__(self, nameFile):
        self.predictions = []
        with open(nameFile) as f:
            tempSuperDict = json.load(f)
        # print(tempSuperDict)
        for prediction in tempSuperDict.values():
            # print(prediction)
            self.predictions.append(Prediction(prediction["typeLocalisation"], prediction["typeTdA"]))
            self.predictions[-1].addData(prediction["data"])
            
            
    def showData(self):
        for pred in self.predictions:
            print(pred.data)
            
            
    def compareData(self, typeLocTab, typeTdATab):
        length = len(typeLocTab)
        sortedData = [] 
        for i in range(0,length):
            sortedData.append([])
            for pred in self.predictions:
                if pred.typeLocalisation == typeLocTab[i] and pred.typeTdA == typeTdATab[i]:
                    sortedData[i].extend(pred.data)
        for i in range(0,length):
            print("Voici toutes les données de type " + typeLocTab[i] + " et " + typeTdATab[i])
            # print(sortedData[i])
        handles = []
        handlesLabel = []
        medianData = []
        ecartData = []    
        for i in range(0,length):
            medianData.append(stat.median(sortedData[i]))
            ecartData.append(stat.pstdev(sortedData[i]))
        for i in range(0,length):
            handles.append(plt.scatter(stat.median(sortedData[i]),stat.pstdev(sortedData[i]),label=typeTdATab[i]))
            handlesLabel.append(typeLocTab[i] + " " + typeTdATab[i])
        plt.legend(handles,handlesLabel)
        plt.ylabel("Écart-type de la norme de l'erreur")
        plt.xlabel("Médiane de la norme de l'erreur")
        plt.xlim(0, max(medianData)+ 1)
        plt.ylim(0, max(ecartData)+ 1)
        print(medianData)
        print(ecartData)

#######################################################
#                    Fonctions
#######################################################

######### Fonctions intermédiaires

def findPeak(tab): # Implémentation naive pour trouver le TdA
    ss = len(tab)
    for i in range(ss):
        if abs(tab[i]) > VALEUR_SEUIL: # À mettre à valeur positive
            return i

def tij(first,second): # Différence de temps d'arrivée
    # return (first.t - second.t)*0.001 # Mettre des secondes au lieu des indices de tableau    
    return (findPeak(first.t) - findPeak(second.t))

def di(source,sensor): # Norme entre 2 points
    return math.sqrt(math.pow((sensor.x-source.x),2)+math.pow((sensor.y-source.y),2)) # Vérifier le calcul et les valeurs

######### Fonctions de calcul de TdA 

def crossCorrelation(signal1,signal2):
    result = sc.signal.correlate(signal1, signal2,mode='full', method='auto') # Fait glisser le premier signal sur le deuxième en partant de la droite du tableau
    return np.argmax(result) - len(signal1) # Le deltaT en indice
    
def hilbertEnveloppe(signal1):
    hilbertTransform = sc.signal.hilbert(signal1)
    enveloppe = np.abs(hilbertTransform)
    return enveloppe
    # Et il faut mettre un seuil sur cette enveloppe. Là encore, à voir ...
      
######### Fonctions de localisation

def trilaterationMethodSeuilNaif(coordonates): # Fonction à minimiser tirée de la revue de Kundu et al.
    Imu9 = IMU(coordonates[0], coordonates[1], 0)
    toRet = 0
    for i in range(0, n-1):
        for j in range(i, n):
            for k in range(0, n-1):
                for l in range(k,n):
                    toRet += math.pow(tij(Tab[i],Tab[j])*(di(Imu9, Tab[k]) - di(Imu9, Tab[l])) - tij(Tab[k],Tab[l])*(di(Imu9, Tab[i]) - di(Imu9, Tab[j])) ,2)
    return toRet

def trilaterationMethodCrossCorrelation(coordonates): # Fonction à minimiser tirée de la revue de Kundu et al.
    Imu9 = IMU(coordonates[0], coordonates[1], 0)
    toRet = 0
    for i in range(0, n-1):
        for j in range(i, n):
            for k in range(0, n-1):
                for l in range(k,n):
                    toRet += math.pow(crossCorrelation(Tab[i].t,Tab[j].t)*(di(Imu9, Tab[k].t) - di(Imu9, Tab[l])) - crossCorrelation(Tab[k].t,Tab[l].t)*(di(Imu9, Tab[i]) - di(Imu9, Tab[j])) ,2)
    return toRet

def trilaterationMethodSeuilEnveloppe(coordonates): # Fonction à minimiser tirée de la revue de Kundu et al.
    Imu9 = IMU(coordonates[0], coordonates[1], 0)
    toRet = 0
    for i in range(0, n-1):
        for j in range(i, n):
            for k in range(0, n-1):
                for l in range(k,n):
                    toRet += math.pow(tij(hilbertEnveloppe(Tab[i].t),hilbertEnveloppe(Tab[j].t))*(di(Imu9, Tab[k]) - di(Imu9, Tab[l])) - tij(hilbertEnveloppe(Tab[k].t),hilbertEnveloppe(Tab[l].t))*(di(Imu9, Tab[i]) - di(Imu9, Tab[j])) ,2)
    return toRet

def trilaterationMethodTransforméeOndelette(coordonates): # Fonction à minimiser tirée de la revue de Kundu et al.
    Imu9 = IMU(coordonates[0], coordonates[1], 0)
    toRet = 0
    for i in range(0, n-1):
        for j in range(i, n):
            for k in range(0, n-1):
                for l in range(k,n):
                    toRet += math.pow(tij(Tab[i],Tab[j])*(di(Imu9, Tab[k]) - di(Imu9, Tab[l])) - tij(Tab[k],Tab[l])*(di(Imu9, Tab[i]) - di(Imu9, Tab[j])) ,2)
    return toRet

 ######### Fonctions autres

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
    Tab = [Imu1,Imu2,Imu3,Imu4,Imu5,Imu6,Imu7,Imu8]

def plotPoints(knownPoint,foundPoint,i): 
    plt.plot(foundPoint[1],foundPoint[0],'rx',label=f"found point {i}")
    plt.plot(knownPoint[1],knownPoint[0],'bx',label="known point")
    plt.ylim(-0.1,1.9)
    plt.xlim(-0.1,1.9)
    plt.legend(loc="upper left")
    plt.show()
    
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
# TYPE_TDA 

    match DATA_SET:
        case "SautStage":
            print("Pas implémenté.")
        case "ImpactStage":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/impacteur_accelero.npy"),np.load("Data/impacteur_localisation.npy"),np.load("Data/impacteur_pos_accelero.npy"))
        case "ToutStage":
            print("Pas implémenté.")
        case "SautMiniProj":
            print("Pas implémenté.")
        case "ImpactMiniProj":
            print("Pas implémenté.")
        case "ToutMiniProj":
            print("Pas implémenté.")
        case "Tout":
            print("Pas implémenté.")
        case _:
            print("Ce dataset n'est pas valable.") 
    
    allNormErrors = []
    x0 = [1,1] # On le centre par rapport à notre tapis, bien que la différence de résultat ne soit a priori pas significative
    b=((-0.1,1.9), (-0.1,1.9))
    
    for current_impact_index in range(0,10): # Pour le pic en valeur absolue, ça donne une valeur absurde pour 112
        if current_impact_index != 112:
            initialize_IMU(ImpactAccelero[current_impact_index],IMULocalisations[current_impact_index],TRAITEMENT_ACCELEROMETRE)
            match TYPE_OPTIMISATION:
                case "Default":
                    match TYPE_LOCALISATION:
                        case "Trilateration" :
                            match TYPE_TDA:
                                case "SeuilNaif" :
                                    res = sc.optimize.minimize(trilaterationMethodSeuilNaif, x0, bounds=b)
                                case "CrossCorrelation" :
                                    res = sc.optimize.minimize(trilaterationMethodCrossCorrelation, x0, bounds=b)
                                case "SeuilEnveloppe" :
                                    res = sc.optimize.minimize(trilaterationMethodSeuilEnveloppe, x0, bounds=b)
                                case "NeuralNetwork" :
                                    print("Not yet ... If ever")   
                                case _ :
                                    print("Cette méthode de calcul du TDA n'est pas valable.")
                        case "NeuralNetwork" :
                            print("Not yet ... If ever")
                        case _ :
                            print("Cette méthode de localisation n'est pas valable.")
                case "Nelder-Mead" | "Powell" | "CG" | "BFGS" | "Newton-CG" | "L-BFGS-B" | "TNC" | "COBYLA" | "SLSQP" | "trust-constr" | "dogleg" | "trust-ncg" | "trust-exact" | "trust-krylov" :
                    match TYPE_LOCALISATION:
                        case "Trilateration" :
                            match TYPE_TDA:
                                case "SeuilNaif" :
                                    res = sc.optimize.minimize(trilaterationMethodSeuilNaif, x0, bounds=b)
                                case "CrossCorrelation" :
                                    res = sc.optimize.minimize(trilaterationMethodCrossCorrelation, x0, bounds=b)
                                case "SeuilEnveloppe" :
                                    res = sc.optimize.minimize(trilaterationMethodSeuilEnveloppe, x0, bounds=b)
                                case "NeuralNetwork" :
                                    print("Not yet ... If ever")   
                                case _ :
                                    print("Cette méthode de calcul du TDA n'est pas valable.")
                        case "NeuralNetwork" :
                            print("Not yet ... If ever")
                        case _ :
                            print("Cette méthode de localisation n'est pas valable.")
                case _:
                    print("Cette méthode d'optimisation n'est pas valable.")

            foundPoint = res.x
            norm_Error = math.sqrt(math.pow((foundPoint[0]-ImpactLocalisation[current_impact_index][0][0]),2)+math.pow((foundPoint[1]-ImpactLocalisation[current_impact_index][1][0]),2))
            allNormErrors.append(norm_Error)
            plotPoints(ImpactLocalisation[current_impact_index],foundPoint,current_impact_index)
    
 
    # analysis(allNormErrors)
    # pred = Prediction("Trilateration", "CrossCorrelation")
    # pred.addData(allNormErrors)
    # pred.saveToJson()
    # test = dataVisualizer(JSON_FILE)
    # test.showData()
    # test.compareData(["Trilateration","Trilateration"], ["SeuilNaif","CrossCorrelation"])
  


if __name__ == "__main__":
    main()






