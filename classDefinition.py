JSON_FILE = "data.json"

import json
import statistics as stat
import matplotlib.pyplot as plt

class IMU:
    def __init__(self, x, y,t):
        self.x = x
        self.y = y
        self.t = t


class Prediction:
    
    
    # On donne ici ce qui définit une prédiction, c'est-à-dire l'ensemble
    # des paramètres sur lesquels on peut jouer.
    # typeLocalisation : "Trilateration" , "CNN"
    # typeTdA : "SeuilNaif" , "CrossCorrelation" , "SeuilEnveloppe" , "TransforméeOndelette"
    # typeOptimisation : "Nelder-Mead" , "" 
    # valeurSeuil : int
    # traitementAccelerometre : "AxeZ" , "Norme" 
    # dataSet : "SautStage" , "ImpactStage", "ToutStage" , "SautMiniProj" , "ImpactMiniProj" , "ToutMiniProj" , "Tout"
    
    
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
        plt.ylabel('Écart-type')
        plt.xlabel('Médiane')
        plt.xlim(0, max(medianData)+ 1)
        plt.ylim(0, max(ecartData)+ 1)
        print(medianData)
        print(ecartData)
