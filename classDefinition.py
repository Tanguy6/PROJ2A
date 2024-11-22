JSON_FILE = "data.json"

import json
import statistics as stat
import matplotlib.pyplot as plt
import scipy as sc
import math
import numpy as np


# Doit être placé là pour éviter un problème d'inclusion circulaire, i.e. 
# un problème problématique.

class IMU:
    def __init__(self, x, y,t):
        self.x = x
        self.y = y
        self.t = t

from peakMethods import tij
from peakMethods import crossCorrelation
from peakMethods import hilbertEnveloppe
from peakMethods import di


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


# Cette classe encapsule les attributs qui permettent de paramétriser une 
# recherche de point (les même paramètres que Prediction) ainsi que les 
# méthodes à appeler pour que le main soit transparent peu importe 
# les paramètres utilisés.
class Finder:
    
    def __init__(self,typeLocalisationParam, typeTdAParam, typeOptimisationParam, valeurSeuilParam, traitementAccelerometreParam, dataSetParam):
        self.typeLocalisation =  typeLocalisationParam
        self.typeTdA = typeTdAParam
        self.typeOptimisation = typeOptimisationParam 
        self.valeurSeuil = valeurSeuilParam 
        self.traitementAccelerometre =  traitementAccelerometreParam
        self.dataSet = dataSetParam
        self.IMUS = [IMU(0,0,0),IMU(0,1,0),IMU(0,2,0),IMU(1,2,0),IMU(2,2,0),IMU(2,1,0),IMU(2,0,0),IMU(1,0,0)]
        # self.tableauImus = tableauImusParam


    def getTdA(self,first,second):
        match self.typeTdA:
            case "SeuilNaif":
                return tij(first,second,self.valeurSeuil)
            case "CrossCorrelation":
                return crossCorrelation(first.t, second.t)
            case "SeuilEnveloppe":
                return hilbertEnveloppe(first.t) - hilbertEnveloppe(second.t) 
            case "TransforméeOndelette":
                print("Pas implémenté.")
            case _:
                print("Cette méthode n'est pas valable.")    
     
        
    def initialize_IMU(self,index):  
        match self.traitementAccelerometre:
            case "AxeZ":
                for i in range(0,8):
                    self.IMUS[i].t = self.ImpactAccelero[index][1+3*i]
            case "Norme":
                # On récupère la norme du vecteur accélération
                m11 = np.transpose(np.linalg.norm(self.ImpactAccelero[index][0:3,:],axis=0))
                m12 = np.transpose(np.linalg.norm(self.ImpactAccelero[index][3:6,:],axis=0))
                m13 = np.transpose(np.linalg.norm(self.ImpactAccelero[index][6:9,:],axis=0))
                m21 = np.transpose(np.linalg.norm(self.ImpactAccelero[index][9:12,:],axis=0))
                m23 = np.transpose(np.linalg.norm(self.ImpactAccelero[index][12:15,:],axis=0))
                m31 = np.transpose(np.linalg.norm(self.ImpactAccelero[index][15:18,:],axis=0))
                m32 = np.transpose(np.linalg.norm(self.ImpactAccelero[index][18:21,:],axis=0))
                m33 = np.transpose(np.linalg.norm(self.ImpactAccelero[index][21:24,:],axis=0))
                 
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
                    
                self.IMUS[0].t = M11
                self.IMUS[1].t = M12
                self.IMUS[2].t = M13
                self.IMUS[3].t = M21
                self.IMUS[4].t = M23
                self.IMUS[5].t = M31
                self.IMUS[6].t = M32
                self.IMUS[7].t = M33  
            case _:
                print("Cette méthode pour récupérer l'accélération n'est pas valable.")
                # Initialisation des positions en x
                self.IMUS[0].x = self.ImpactLocalisation[index][0][0]
                self.IMUS[1].x = self.ImpactLocalisation[index][1][0]
                self.IMUS[2].x = self.ImpactLocalisation[index][2][0]
                self.IMUS[3].x = self.ImpactLocalisation[index][3][0]
                self.IMUS[4].x = self.ImpactLocalisation[index][4][0]
                self.IMUS[5].x = self.ImpactLocalisation[index][5][0]
                self.IMUS[6].x = self.ImpactLocalisation[index][6][0]
                self.IMUS[7].x = self.ImpactLocalisation[index][7][0]
                # Initialisation des positions en y
                self.IMUS[0].y = self.ImpactLocalisation[index][0][1]
                self.IMUS[1].y = self.ImpactLocalisation[index][1][1]
                self.IMUS[2].y = self.ImpactLocalisation[index][2][1]
                self.IMUS[3].y = self.ImpactLocalisation[index][3][1]
                self.IMUS[4].y = self.ImpactLocalisation[index][4][1]
                self.IMUS[5].y = self.ImpactLocalisation[index][5][1]
                self.IMUS[6].y = self.ImpactLocalisation[index][6][1]
                self.IMUS[7].y = self.ImpactLocalisation[index][7][1]
        
     
    def trilaterationMethod(self,coordonates): # Fonction à minimiser tirée de la revue de Kundu et al.
        n = 8
        Imu9 = IMU(coordonates[0], coordonates[1], 0)
        toRet = 0
        for i in range(0, n-1):
            for j in range(i, n):
                for k in range(0, n-1):
                    for l in range(k,n):
                        # print(i)
                        toRet += math.pow(tij(self.IMUS[i],self.IMUS[j],self.valeurSeuil)*(di(Imu9, self.IMUS[k]) - di(Imu9, self.IMUS[l])) - tij(self.IMUS[k],self.IMUS[l],self.valeurSeuil)*(di(Imu9, self.IMUS[i]) - di(Imu9, self.IMUS[j])) ,2)
        return toRet
     
        
    
    def getRealPoint(self,index):
        return self.ImpactLocalisation[index]
    
    def chargerDataSet(self):
        match self.dataSet:
            case "SautStage":
                print("Pas implémenté.")
            case "ImpactStage":
                (self.ImpactAccelero, self.ImpactLocalisation, self.IMULocalisations) = (np.load("Data/impacteur_accelero.npy"),np.load("Data/impacteur_localisation.npy"),np.load("Data/impacteur_pos_accelero.npy"))
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
     
    def getPredictedPoint(self):
        from isotropicUnknownCelerityTraingulation import trilaterationMethod
        x0 = [1,1] # On le centre par rapport à notre tapis, bien que la différence de résultat ne soit a priori pas significative
        b=((-0.1,1.9), (-0.1,1.9))
        match self.typeOptimisation:
            case "Default":
                match self.typeLocalisation:
                    case "Trilateration" :
                        res = sc.optimize.minimize(self.trilaterationMethod, x0, bounds=b)
                        return res.x
                    case "NeuralNetwork" :
                        print("")
                    case _ :
                        print("Cette méthode de localisation n'est pas valable.")
            case "Nelder-Mead" | "Powell" | "CG" | "BFGS" | "Newton-CG" | "L-BFGS-B" | "TNC" | "COBYLA" | "SLSQP" | "trust-constr" | "dogleg" | "trust-ncg" | "trust-exact" | "trust-krylov" :
                match self.typeLocalisation:
                    case "Trilateration" :
                        res = sc.optimize.minimize(self.trilaterationMethod, x0, method=self.typeOptimisation, bounds=b)
                        return res.x
                    case "NeuralNetwork" :
                        print("")
                    case _ :
                        print("Cette méthode de localisation n'est pas valable.")
            case _:
                print("Cette méthode d'optimisation n'est pas valable.")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        