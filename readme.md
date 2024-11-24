# Projet de recherche : Localisation de source d'impact par triangulation pour l'analyse des réceptions de saut en gymnastique artistique.

## Encadré par Suzon PUCHEU

## Principe :

L'objectif est d'essayer plusieurs méthodes trouvées dans la litérature afin de localiser un impact sur un tapis de gymnastique. Nous avons à notre disposition des données d'accéléromètre placés sur un tapis sur lequel ont été réalisés des impacts. 

## Fiche de route :

Voici la découpe des différents objectifs.

### Méthodes de localisation :

- [x] Trilatération de Kundu et al.
  
  *À compléter* 

### Méthodes de calcul des différences des temps d'arrivée

- [x] Seuil naïf

- [x] Cross-corrélation

- [ ] Seuil à enveloppe

- [ ] Transformée en ondelettes

### Traitement des données

- [x] Calcul de valeurs statistiques (Moyenne, Médiane, Écart-type)

- [x] Sauvegarde des données dans un fichier pour traiter les différences entre méthodes

- [x] Affichage des différences en médiane et écart-type des différentes méthodes

- [ ] Utilisation ergonomique de l'affichage

- [ ] Vérification de la similitude entre donées par un test statistique

### Intégration

- [X] Intégrer les différentes méthodes au sein d'un seul code cohérent et simple d'usage

## Paramètres de réglage implémentés

- typeLocalisation
  - [X]  Trilateration
- typeTdA
  - [X] SeuilNaif
  - [ ] CrossCorrelation
  - [ ] SeuilEnveloppe
  - [ ] TransforméeOndelette
- typeOptimisation
  - [X] Nelder-Mead
- valeurSeuil
  - [X] int
  - [ ] Adaptatif ?
- TraitementAccelerometre
  - [X] AxeZ
  - [X] Norme
- dataSet
  - [ ] SautStage
  - [X] ImpactStage
  - [ ] ToutStage
  - [ ] SautMiniProjet
  - [ ] ImpactMiniProjet
  - [ ] ToutMiniProjet
  - [ ] Tout

Ce document est voué à être modifié au fur et à mesure.
