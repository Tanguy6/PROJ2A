JSON_FILE = "data.json"

import json
import statistics as stat
import matplotlib.pyplot as plt
import scipy as sc

from peakMethods import tij
from peakMethods import crossCorrelation
from peakMethods import hilbertEnveloppe
from peakMethods import trilaterationMethod



class IMU:
    def __init__(self, x, y,t):
        self.x = x
        self.y = y
        self.t = t

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
    
    def __init__(self,typeLocalisationParam, typeTdAParam, typeOptimisationParam, valeurSeuilParam, traitementAccelerometreParam, dataSetParam, tableauImusParam):
        self.typeLocalisation =  typeLocalisationParam
        self.typeTdA = typeTdAParam
        self.typeOptimisation = typeOptimisationParam 
        self.valeurSeuil = valeurSeuilParam 
        self.traitementAccelerometre =  traitementAccelerometreParam
        self.dataSet = dataSetParam
        self.tableauImus = tableauImusParam


    def getTdA(self,first,second):
        match self.typeTdA:
            case "SeuilNaif":
                return tij(first,second)
            case "CrossCorrelation":
                return crossCorrelation(first.t, second.t)
            case "SeuilEnveloppe":
                return hilbertEnveloppe(first.t) - hilbertEnveloppe(second.t) 
            case "TransforméeOndelette":
                print("Pas implémenté.")
            case _:
                print("Cette méthode n'est pas valable.")    
     
        
     
    def getPredictedPoint(self):
        x0 = [1,1] # On le centre par rapport à notre tapis, bien que la différence de résultat ne soit a priori pas significative
        match self.typeOptimisation:
            case "Default":
                match self.typeLocalisation:
                    case "Trilateration" :
                        res = sc.optimize.minimize(trilaterationMethod, x0, tol=1e-6)
                        return res.x
                    case "NeuralNetwork" :
                        print("")
                    case _ :
                        print("Cette méthode de localisation n'est pas valable.")
            case "Nelder-Mead" | "Powell" | "CG" | "BFGS" | "Newton-CG" | "L-BFGS-B" | "TNC" | "COBYLA" | "SLSQP" | "trust-constr" | "dogleg" | "trust-ncg" | "trust-exact" | "trust-krylov" :
                match self.typeLocalisation:
                    case "Trilateration" :
                        res = sc.optimize.minimize(trilaterationMethod, x0, method=self.typeOptimisation, tol=1e-6)
                        return res.x
                    case "NeuralNetwork" :
                        print("")
                    case _ :
                        print("Cette méthode de localisation n'est pas valable.")
            case _:
                print("Cette méthode d'optimisation n'est pas valable.")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        