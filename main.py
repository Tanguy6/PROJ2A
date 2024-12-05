
# Tout est rassemble dans un seul code pour accelerer l'execution, par rapport
# a l'optimisation qui fait beaucoup d'appel fonction.


#######################################################
#                    Import
#######################################################

import scipy as sc
import scipy.stats as scstats
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import json
import scikit_posthocs as sp
import difflib as dfl
import sys

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
VALEUR_SEUIL = 4
# TRAITEMENT_ACCELEROMETRE : "AxeZ" , "Norme" 
TRAITEMENT_ACCELEROMETRE = "Norme"
# DATA_SET : "SurtapisSautStage" , "SurtapisImpactStage", "SurtapisToutStage" , 
# "SurtapisSautMiniProj" , "SurtapisImpactMiniProj" , "SurtapisToutMiniProj" , 
# "SurtapisTout", "TapisSautStage" , "TapisImpactStage", "TapisToutStage" , 
# "TapisSautMiniProj" , "TapisImpactMiniProj" , "TapisToutMiniProj" , "TapisTout"
# "TapisStatiqueSautMiniProj" , "TapisStatiqueImpactMiniProj" , "SurtapisStatiqueImpactMiniProj" 
DATA_SET = "TapisSautStage"
SHOW_FIGURES = True
IMAGE_PATH = ""




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
    # dataSet : "SurtapisSautStage" , "SurtapisImpactStage", "SurtapisToutStage" , 
    # "SurtapisSautMiniProj" , "SurtapisImpactMiniProj" , "SurtapisToutMiniProj" , 
    # "SurtapisTout", "TapisSautStage" , "TapisImpactStage", "TapisToutStage" , 
    # "TapisSautMiniProj" , "TapisImpactMiniProj" , "TapisToutMiniProj" , "TapisTout"
    # "TapisStatiqueSautMiniProj" , "TapisStatiqueImpactMiniProj" , "SurtapisStatiqueImpactMiniProj" 
    
    
    def __init__(self, paramTypeLocalisation, paramTypeTdA,paramTypeOptimisation, paramValeurSeuil, paramTraitementAccelerometre, paramDataSet):
        self.typeLocalisation = paramTypeLocalisation
        self.typeTdA = paramTypeTdA
        self.typeOptimisation = paramTypeOptimisation
        self.valeurSeuil = paramValeurSeuil
        self.traitementAccelerometre = paramTraitementAccelerometre
        self.dataSet = paramDataSet
        
    def addData(self,paramData):
        self.data = paramData
        
    def addDataRatio(self,paramDataRatio):
        self.dataRatio = paramDataRatio
        
    def ratioVsError(self):
        plt.scatter(self.data,self.dataRatio)
        plt.xlabel("Erreur en norme")
        plt.ylabel("Ratio de ressemblance")
        if SHOW_FIGURES:
            plt.show()
        else:
            plt.savefig(IMAGE_PATH + '/ratioVerror.png',bbox_inches='tight', dpi = 500 )
            plt.close()
    
    def saveToJson(self):
        tempSuperDict = {}
        with open(JSON_FILE) as f:
            tempSuperDict = json.load(f)
        tempDict = {}
        tempDict["typeLocalisation"] = self.typeLocalisation
        tempDict["typeTdA"] = self.typeTdA
        tempDict["typeOptimisation"] = self.typeOptimisation
        tempDict["valeurSeuil"] = self.valeurSeuil
        tempDict["traitementAccelerometre"] = self.traitementAccelerometre
        tempDict["dataSet"] = self.dataSet
        tempDict["data"] = self.data
        tempDict["dataRatio"] = self.dataRatio
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
            self.predictions.append(Prediction(prediction["typeLocalisation"], prediction["typeTdA"], prediction["typeOptimisation"], prediction["valeurSeuil"], prediction["traitementAccelerometre"], prediction["dataSet"]))
            self.predictions[-1].addData(prediction["data"])
            self.predictions[-1].addData(prediction["dataRatio"])
            
            
    def showData(self):
        for pred in self.predictions:
            print(pred.data)
            
            
            
    def isTwoPopulationstatisticallyDifferent(self, pop1, pop2):
        # Moyennes
        moyennePop1 = stat.mean(pop1)
        moyennePop2 = stat.mean(pop2)
        # Ecart-type
        ecarttype1 = stat.stdev(pop1)
        ecarttype2 = stat.stdev(pop2)
        
        # Nous allons realiser un Z test puisque nous avons plus de 30 samples
        # voir la diapo 17 de ANOVA
        
        Z = (moyennePop1-moyennePop2)/(math.sqrt( ecarttype1 /len(pop1)+ecarttype2/len(pop2)))
        
        print(abs(Z))   
        
        
        # On prend alpha = 0.05, et donc si abs(Z) plus grand que 1,96, on 
        # rejete H0 (hypothese selon laquelle les populations sont les "memes")
        # avec 5 % de conficance qu'elle soit juste sinon on l'accepte
        
        return abs(Z) > 1.96
    
    
    
    def anovaTest(self, typeLocTab, typeTdATab, typeOptimisationTab, traitementAccelerometreTab, dataSetTab):
        # Nous allons faire un ANOVA pour toutes les données dont les paramètres 
        # sont passés en paramètre, comme pour compaData.
        
        # Nous commençons par creer nos dataset tries
            
        length = len(typeLocTab)
        if not len(typeLocTab) == len(typeTdATab) == len(typeOptimisationTab) == len(traitementAccelerometreTab) == len(dataSetTab):
            print("Tout les tableaux doivent avoir la même taille.")
            return -1
        
        
        sortedData = [] 
        for i in range(0,length):
            sortedData.append([])
            for pred in self.predictions:
                if pred.typeLocalisation == typeLocTab[i] and pred.typeTdA == typeTdATab[i] and pred.typeOptimisation == typeOptimisationTab[i] and pred.traitementAccelerometre == traitementAccelerometreTab[i] and pred.dataSet == dataSetTab[i]:
                    sortedData[i].extend(pred.data)
        # for i in range(0,length):
            # print("Voici toutes les données de type " + typeLocTab[i] + " et " + typeTdATab[i] + " et " + typeOptimisationTab[i] + " et " + traitementAccelerometreTab[i] + " et " + dataSetTab[i])
            # print(sortedData[i])

        
        
        # On teste la normalité de chaque échantillon
        # Si la p-value est supérieure à notre seuil de confiance (0,05)
        # nous ne pouvons pas rejeter l'hypothèse nul donc les données
        # sont normales
        
        flagNormality = True
        
        for i in range(0,length):
            if(scstats.shapiro(sortedData[i])[1] < 0.05):
                print("Un des échantillons ne suit pas une loi normale. \nNous passons alors par le test de Kruskal Wallis.")
                flagNormality = False
                break
        
        if (flagNormality):
            
            # Une fois que nous sommes surs que nos echantillons sont normaux,
            # nous faisons notre one way anova
            if (scstats.f_oneway(*sortedData)[1] > 0.05):
                print("On peut accepter l'hypothèse nulle avec une confiance de 5%.")
                return
                
            print("L'hypothèse nulle est rejetée, donc il y a une différence significative entre les echantillons.")    
            
            # S'il y a une différence statistiquement significative, on utilise un test post hoc pour savoir quel 
            # couple pose souci
            
            res = scstats.tukey_hsd(*sortedData)
            
            
            for i in range (0,length):
                for j in range (i+1,length):
                    # print("(" + str(i) + ";" + str(j) + ")")
                    if res.pvalue[i,j] < 0.05: # S'ils sont significativement differents
                        print("Les groupes " + str(i) + " et " + str(j) + " sont significativement différents.")
        else:
            
            # Si la loi suivie n'est pas normale, il faut alors réaliser un test
            # de Kruskal Wallis, suivi d'un test post-hoc de Dunn
            
            if (scstats.kruskal(*sortedData)[1] > 0.05):
                print("On peut accepter l'hypothèse nulle avec une confiance de 5%.")
                return 
                
            print("L'hypothèse nulle est rejetée, donc il y a une différence significative entre les echantillons.")    
            
            # S'il y a une différence statistiquement significative, on utilise un test post hoc pour savoir quel 
            # couple pose souci
            
            p_values = sp.posthoc_dunn(sortedData, p_adjust='holm')

            
            for i in range (0,length):
                for j in range (i+1,length):
                    # print("(" + str(i) + ";" + str(j) + ")")
                    if p_values.loc[i+1,j+1] < 0.05: # S'ils sont significativement differents
                        print("Les groupes " + str(i) + " et " + str(j) + " sont significativement différents.")
            
        
            
    def compareData(self, typeLocTab, typeTdATab, typeOptimisationTab, traitementAccelerometreTab, dataSetTab):
        length = len(typeLocTab)
        sortedData = [] 
        for i in range(0,length):
            sortedData.append([])
            for pred in self.predictions:
                if pred.typeLocalisation == typeLocTab[i] and pred.typeTdA == typeTdATab[i] and pred.typeOptimisation == typeOptimisationTab[i] and pred.traitementAccelerometre == traitementAccelerometreTab[i] and pred.dataSet == dataSetTab[i]:
                    sortedData[i].extend(pred.data)
        for i in range(0,length):
            print("Voici toutes les données de type " + typeLocTab[i] + " et " + typeTdATab[i] + " et " + typeOptimisationTab[i] + " et " + traitementAccelerometreTab[i] + " et " + dataSetTab[i])
            # print(sortedData[i])
        handles = []
        handlesLabel = []
        meanData = []
        ecartData = []    
        for i in range(0,length):
            meanData.append(stat.mean(sortedData[i]))
            ecartData.append(stat.pstdev(sortedData[i]))
        for i in range(0,length):
            handles.append(plt.scatter(stat.mean(sortedData[i]),stat.pstdev(sortedData[i]),label=typeTdATab[i],marker = 'x' ))
            handlesLabel.append(typeLocTab[i] + " " + typeTdATab[i] + " " + typeOptimisationTab[i] + " " + traitementAccelerometreTab[i] + " " + dataSetTab[i])
        plt.legend(handles,handlesLabel)
        plt.ylabel("Écart-type de la norme de l'erreur")
        plt.xlabel("Moyenne de la norme de l'erreur")
        plt.xlim(0, max(meanData)+ 1)
        plt.ylim(0, max(ecartData)+ 1)
        print(meanData)
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
    # Si on met le findPeak ici on ralentit (pas autant que l'autre mais quand meme)
    return first.t - second.t

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
    Tab = [Imu1,Imu2,Imu3,Imu4,Imu5,Imu6,Imu7,Imu8]
    Imu9 = IMU(coordonates[0], coordonates[1], 0)
    toRet = 0
    for i in range(0, n-1):
        for j in range(i+1, n):
            for k in range(0, n-1):
                for l in range(k+1,n):
                    toRet += math.pow(  tij(Tab[i],Tab[j])*(di(Imu9, Tab[k]) - di(Imu9, Tab[l])) - tij(Tab[k],Tab[l])*(di(Imu9, Tab[i]) - di(Imu9, Tab[j])) ,2)
    return toRet

def trilaterationMethodCrossCorrelation(coordonates): 
    Tab = [Imu1,Imu2,Imu3,Imu4,Imu5,Imu6,Imu7,Imu8]
    Imu9 = IMU(coordonates[0], coordonates[1], 0)
    toRet = 0
    for i in range(0, n-1):
        for j in range(i, n):
            for k in range(0, n-1):
                for l in range(k,n):
                    toRet += math.pow(crossCorrelation(Tab[i].t,Tab[j].t)*(di(Imu9, Tab[k]) - di(Imu9, Tab[l])) - crossCorrelation(Tab[k].t,Tab[l].t)*(di(Imu9, Tab[i]) - di(Imu9, Tab[j])) ,2)
    return toRet

def trilaterationMethodSeuilEnveloppe(coordonates): # Fonction à minimiser tirée de la revue de Kundu et al.
    Tab = [Imu1,Imu2,Imu3,Imu4,Imu5,Imu6,Imu7,Imu8]
    Imu9 = IMU(coordonates[0], coordonates[1], 0)
    toRet = 0
    for i in range(0, n-1):
        for j in range(i, n):
            for k in range(0, n-1):
                for l in range(k,n):
                    toRet += math.pow(tij(Tab[i],Tab[j])*(di(Imu9, Tab[k]) - di(Imu9, Tab[l])) - tij(Tab[k],Tab[l])*(di(Imu9, Tab[i]) - di(Imu9, Tab[j])) ,2)
    return toRet

def trilaterationMethodTransforméeOndelette(coordonates): # Fonction à minimiser tirée de la revue de Kundu et al.
    Tab = [Imu1,Imu2,Imu3,Imu4,Imu5,Imu6,Imu7,Imu8]
    Imu9 = IMU(coordonates[0], coordonates[1], 0)
    toRet = 0
    for i in range(0, n-1):
        for j in range(i, n):
            for k in range(0, n-1):
                for l in range(k,n):
                    toRet += math.pow(tij(Tab[i],Tab[j])*(di(Imu9, Tab[k]) - di(Imu9, Tab[l])) - tij(Tab[k],Tab[l])*(di(Imu9, Tab[i]) - di(Imu9, Tab[j])) ,2)
    return toRet

######### Fonctions autres

def initialize_IMU_Temporel(CurrentImpactAccelero,traitementAccelerometreParam):  
    match traitementAccelerometreParam:
        case "AxeZ":
            Imu1.t = findPeak(CurrentImpactAccelero[2])
            Imu2.t = findPeak(CurrentImpactAccelero[5])
            Imu3.t = findPeak(CurrentImpactAccelero[8])
            Imu4.t = findPeak(CurrentImpactAccelero[11])
            Imu5.t = findPeak(CurrentImpactAccelero[14])
            Imu6.t = findPeak(CurrentImpactAccelero[17])
            Imu7.t = findPeak(CurrentImpactAccelero[20])
            Imu8.t = findPeak(CurrentImpactAccelero[23])
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
    
            match TYPE_TDA:
                case "SeuilNaif": 
                    Imu1.t = findPeak(M11)
                    Imu2.t = findPeak(M12)
                    Imu3.t = findPeak(M13)
                    Imu4.t = findPeak(M21)
                    Imu5.t = findPeak(M23)
                    Imu6.t = findPeak(M31)
                    Imu7.t = findPeak(M32)
                    Imu8.t = findPeak(M33)
                case "CrossCorrelation": 
                    Imu1.t = M11
                    Imu2.t = M12
                    Imu3.t = M13
                    Imu4.t = M21
                    Imu5.t = M23
                    Imu6.t = M31
                    Imu7.t = M32
                    Imu8.t = M33
                case "SeuilEnveloppe":
                    Imu1.t = findPeak(hilbertEnveloppe(M11))
                    Imu2.t = findPeak(hilbertEnveloppe(M12))
                    Imu3.t = findPeak(hilbertEnveloppe(M13))
                    Imu4.t = findPeak(hilbertEnveloppe(M21))
                    Imu5.t = findPeak(hilbertEnveloppe(M23))
                    Imu6.t = findPeak(hilbertEnveloppe(M31))
                    Imu7.t = findPeak(hilbertEnveloppe(M32))
                    Imu8.t = findPeak(hilbertEnveloppe(M33))
                case _:
                    print("Cette méthode pour récupérer le TdA n'est pas valable.")
                    
        case _:
            print("Cette méthode pour récupérer l'accélération n'est pas valable.")

def initialize_IMU_Spatial(IMULocalisations):  

    if "Statique" in DATA_SET:    
        # Initialisation des positions en x
        Imu1.x = IMULocalisations[0][0]
        Imu2.x = IMULocalisations[1][0]
        Imu3.x = IMULocalisations[2][0]
        Imu4.x = IMULocalisations[3][0]
        Imu5.x = IMULocalisations[4][0]
        Imu6.x = IMULocalisations[5][0]
        Imu7.x = IMULocalisations[6][0]
        Imu8.x = IMULocalisations[7][0]
        # Initialisation des positions en y
        Imu1.y = IMULocalisations[0][1]
        Imu2.y = IMULocalisations[1][1]
        Imu3.y = IMULocalisations[2][1]
        Imu4.y = IMULocalisations[3][1]
        Imu5.y = IMULocalisations[4][1]
        Imu6.y = IMULocalisations[5][1]
        Imu7.y = IMULocalisations[6][1]
        Imu8.y = IMULocalisations[7][1]
    else:
        # Initialisation des positions en x
        Imu1.x = np.mean(IMULocalisations[:,0,0])
        Imu2.x = np.mean(IMULocalisations[:,1,0])
        Imu3.x = np.mean(IMULocalisations[:,2,0])
        Imu4.x = np.mean(IMULocalisations[:,3,0])
        Imu5.x = np.mean(IMULocalisations[:,4,0])
        Imu6.x = np.mean(IMULocalisations[:,5,0])
        Imu7.x = np.mean(IMULocalisations[:,6,0])
        Imu8.x = np.mean(IMULocalisations[:,7,0])
        # Initialisation des positions en y
        Imu1.y = np.mean(IMULocalisations[:,0,1])
        Imu2.y = np.mean(IMULocalisations[:,1,1])
        Imu3.y = np.mean(IMULocalisations[:,2,1])
        Imu4.y = np.mean(IMULocalisations[:,3,1])
        Imu5.y = np.mean(IMULocalisations[:,4,1])
        Imu6.y = np.mean(IMULocalisations[:,5,1])
        Imu7.y = np.mean(IMULocalisations[:,6,1])
        Imu8.y = np.mean(IMULocalisations[:,7,1])

def plotPoints(knownPoint,foundPoint,i): 
    plt.plot(foundPoint[0],foundPoint[1],'rx',label=f"found point {i}")
    plt.plot(knownPoint[0],knownPoint[1],'bx',label="known point")
    plt.ylim(-0.1,1.9)
    plt.xlim(-0.1,1.9)
    plt.legend(loc="upper left")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.savefig(IMAGE_PATH + f"/plotPoints{i}.png",bbox_inches='tight', dpi = 500 )
        plt.close()
    
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

def findPoint(CurrentImpactAccelero): 
    x0 = [1,1]
    b=((-0.1,1.9), (-0.1,1.9))
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
    return res.x

        
def plotAllFoundAndKnownPoints(foundPoints,KnownPoints):
    
    foundPointsAllX = [item[0] for item in foundPoints]
    foundPointsAllY = [item[1] for item in foundPoints]
    
    knownPointsAllX = [item[0][0] for item in KnownPoints]
    knownPointsAllY = [item[1][0] for item in KnownPoints]
    
    plt.scatter(foundPointsAllX,foundPointsAllY,color = 'r',marker = 'x' )
    plt.scatter(knownPointsAllX,knownPointsAllY,color = 'b',marker = 'x' )
    plt.ylim(-0.1,1.9)
    plt.xlim(-0.1,1.9)
    plt.legend(["Points trouvés","Points connus"],bbox_to_anchor=(1.05, 1.0), loc='upper left')
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.savefig(IMAGE_PATH + '/comparaisonToutPoint.png',bbox_inches='tight', dpi = 500 )
        plt.close()


def differentiateSupposedAndTrueIMUsOrder(knownPointx,knownPointy):
    Tab = [Imu1,Imu2,Imu3,Imu4,Imu5,Imu6,Imu7,Imu8]
    Imu9 = IMU(knownPointx, knownPointy, 0)
    d1 = di(Imu9, Tab[0])
    d2 = di(Imu9, Tab[1])
    d3 = di(Imu9, Tab[2])
    d4 = di(Imu9, Tab[3])
    d5 = di(Imu9, Tab[4])
    d6 = di(Imu9, Tab[5])
    d7 = di(Imu9, Tab[6])
    d8 = di(Imu9, Tab[7])
    
    TabDi = [d1,d2,d3,d4,d5,d6,d7,d8]
    
    ind_tri_distance = np.argsort(TabDi)
    
    match TYPE_TDA:
        case "SeuilNaif" | "SeuilEnveloppe": 
            TabTi = [Imu1.t,Imu2.t,Imu3.t,Imu4.t,Imu5.t,Imu6.t,Imu7.t,Imu8.t]
        case "CrossCorrelation":
            TabTi = [0,crossCorrelation(Imu1.t, Imu2.t),crossCorrelation(Imu1.t, Imu3.t),crossCorrelation(Imu1.t, Imu4.t),crossCorrelation(Imu1.t, Imu5.t),crossCorrelation(Imu1.t, Imu6.t),crossCorrelation(Imu1.t, Imu7.t),crossCorrelation(Imu1.t, Imu8.t)]
        case _:
            TabTi = [0,0,0,0,0,0,0,0]
    
    
    ind_tri_tmps = np.argsort(TabTi)
    
    
    # On les met toujours dans ce sens là car cette bibliothèque est sensible au sens
    # de comparaison.
    
    matcher = dfl.SequenceMatcher(None,ind_tri_distance,ind_tri_tmps)
    
    
    # print("Tri distance")
    # print(ind_tri_distance)
    # print("Tri temps")
    # print(ind_tri_tmps)
    # print("Ratio des deux séquences")
    # print(matcher.ratio())
    # print("-----")    
    return matcher.ratio()


def main():
    match DATA_SET:
        case "SurtapisSautStage":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/surtapis/saut/sauts_accelero_set1.npy"),np.load("Data/surtapis/saut/sauts_localisation_set1.npy"),np.load("Data/surtapis/saut/sauts_pos_accelero_set1.npy"))
        case "SurtapisImpactStage":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/surtapis/impacteur/impacteur_accelero_set1.npy"),np.load("Data/surtapis/impacteur/impacteur_localisation_set1.npy"),np.load("Data/surtapis/impacteur/impacteur_pos_accelero_set1.npy"))
        case "SurtapisToutStage":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/surtapis/complet/data set_accelero_set1.npy"),np.load("Data/surtapis/complet/data set_localisation_set1.npy"),np.load("Data/surtapis/complet/data set_pos_accelero_set1.npy"))
        case "SurtapisSautMiniProj":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/surtapis/saut/sauts_accelero_set2.npy"),np.load("Data/surtapis/saut/sauts_localisation_set2.npy"),np.load("Data/surtapis/saut/sauts_pos_accelero_set2.npy"))
        case "SurtapisImpactMiniProj":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/surtapis/impacteur/impacteur_accelero_set2.npy"),np.load("Data/surtapis/impacteur/impacteur_localisation_set2.npy"),np.load("Data/surtapis/impacteur/impacteur_pos_accelero_set2.npy"))
        case "SurtapisToutMiniProj":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/surtapis/complet/data set_accelero_set2.npy"),np.load("Data/surtapis/complet/data set_localisation_set2.npy"),np.load("Data/surtapis/complet/data set_pos_accelero_set2.npy"))
        case "TapisSautStage":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/tapis/sauts/sauts_accelero_set1.npy"),np.load("Data/tapis/sauts/sauts_localisation_set1.npy"),np.load("Data/tapis/sauts/sauts_pos_accelero_set1.npy"))
        case "TapisImpactStage":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/tapis/impacteur/impacteur_accelero_set1.npy"),np.load("Data/tapis/impacteur/impacteur_localisation_set1.npy"),np.load("Data/tapis/impacteur/impacteur_pos_accelero_set1.npy"))
        case "TapisToutStage":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/tapis/complet/data set_accelero_set1.npy"),np.load("Data/tapis/complet/data set_localisation_set1.npy"),np.load("Data/tapis/complet/data set_pos_accelero_set1.npy"))
        case "TapisSautMiniProj":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/tapis/sauts/sauts_accelero_set2.npy"),np.load("Data/tapis/sauts/sauts_localisation_set2.npy"),np.load("Data/tapis/sauts/sauts_pos_accelero_set2.npy"))
        case "TapisImpactMiniProj":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/tapis/impacteur/impacteur_accelero_set2.npy"),np.load("Data/tapis/impacteur/impacteur_localisation_set2.npy"),np.load("Data/tapis/impacteur/impacteur_pos_accelero_set2.npy"))
        case "TapisToutMiniProj":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/tapis/complet/data set_accelero_set2.npy"),np.load("Data/tapis/complet/data set_localisation_set2.npy"),np.load("Data/tapis/complet/data set_pos_accelero_set2.npy"))
        case "TapisStatiqueSautMiniProj":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/tapis/sauts/sauts_accelero_set2.npy"),np.load("Data/tapis/sauts/sauts_localisation_set2.npy"),np.load("Data/tapis/sauts/statique_tapis.npy"))
        case "TapisStatiqueImpactMiniProj":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/tapis/impacteur/impacteur_accelero_set2.npy"),np.load("Data/tapis/impacteur/impacteur_localisation_set2.npy"),np.load("Data/tapis/impacteur/statique_tapis.npy"))
        case "SurtapisStatiqueImpactMiniProj":
            (ImpactAccelero, ImpactLocalisation, IMULocalisations) = (np.load("Data/surtapis/impacteur/impacteur_accelero_set2.npy"),np.load("Data/surtapis/impacteur/impacteur_localisation_set2.npy"),np.load("Data/surtapis/impacteur/statique_surtapis.npy"))
        case _:
            print("Ce dataset n'est pas valable.") 
            
    nb_impact = len(ImpactAccelero)
    deb = 0
    
    err = []
    ratio = []
    
    foundPoints = []

    # Partie a decommenter pour faire les calculs


    # initialize_IMU_Spatial(IMULocalisations)
    # for current_impact_index in range(deb, deb+nb_impact):
    #     # print(current_impact_index)
    #     initialize_IMU_Temporel(ImpactAccelero[current_impact_index],TRAITEMENT_ACCELEROMETRE)
    #     foundPoint = findPoint(ImpactAccelero[current_impact_index])
    #     norm_Error = math.sqrt(math.pow((foundPoint[0]-ImpactLocalisation[current_impact_index][0][0]),2)+math.pow((foundPoint[1]-ImpactLocalisation[current_impact_index][1][0]),2))
    #     print(current_impact_index)
    #     if norm_Error > 3:
    #         print("Erreur dans le calcul de la norme.")
    #     else :    
    #         plotPoints(ImpactLocalisation[current_impact_index],foundPoint,current_impact_index)
    #         ratio.append(differentiateSupposedAndTrueIMUsOrder(ImpactLocalisation[current_impact_index][0][0],ImpactLocalisation[current_impact_index][1][0]))
    #         err.append(norm_Error)
    #         foundPoints.append([foundPoint[0],foundPoint[1]])
        
        
    # plotAllFoundAndKnownPoints(foundPoints,ImpactLocalisation)
    
    # prediction = Prediction(TYPE_LOCALISATION, TYPE_TDA, TYPE_OPTIMISATION, VALEUR_SEUIL, TRAITEMENT_ACCELEROMETRE, DATA_SET)
    
    # prediction.addData(err)
    
    # prediction.addDataRatio(ratio)
    
    # prediction.saveToJson()

    # Partie a decommenter pour faire de la comparaison
    
    
    dataVisu = dataVisualizer(JSON_FILE)
    # # # print(foundPoints[:][1])
    dataVisu.anovaTest(["Trilateration","Trilateration","Trilateration"], ["SeuilNaif","CrossCorrelation","SeuilEnveloppe"], ["Default","Default","Default"], ["Norme","Norme","Norme"], ["TapisSautStage","TapisSautStage","TapisSautStage"])
    dataVisu.compareData(["Trilateration","Trilateration","Trilateration"], ["SeuilNaif","CrossCorrelation","SeuilEnveloppe"], ["Default","Default","Default"], ["Norme","Norme","Norme"], ["TapisSautStage","TapisSautStage","TapisSautStage"])
    # dataVisu.compareData(["Trilateration","Trilateration"], ["SeuilNaif","CrossCorrelation"])
        
    # print(dataVisu.isTwoPopulationstatisticallyDifferent([1,¨4,5], [1,4,6]))
    

    print('argument list', sys.argv)
    


if __name__ == "__main__":
    t = time.time()
    
    # On réaffecte les constantes avec les arguments passés lors de l'appel, s'il y en a 
    if len(sys.argv) > 3:
        TYPE_LOCALISATION = sys.argv[1]
        TYPE_TDA = sys.argv[2]
        TYPE_OPTIMISATION = sys.argv[3]
        VALEUR_SEUIL = float(sys.argv[4])
        TRAITEMENT_ACCELEROMETRE = sys.argv[5]
        DATA_SET = sys.argv[6]
        SHOW_FIGURES = False
        IMAGE_PATH = "Images/" + TYPE_LOCALISATION + "_" + TYPE_TDA + "_"  + TYPE_OPTIMISATION + "_"  + str(VALEUR_SEUIL) + "_"  + TRAITEMENT_ACCELEROMETRE + "_" + DATA_SET
        

    main()
    print("Temps d'execution': " + str(time.time() - t))