# Projet de recherche : Localisation de source d'impact par triangulation pour l'analyse des réceptions de saut en gymnastique artistique.

## Encadré par Suzon PUCHEU

## Principe :

L'objectif est d'essayer plusieurs méthodes trouvées dans la litérature afin de localiser un impact sur un tapis de gymnastique. Nous avons à notre disposition des données d'accéléromètre placés sur un tapis sur lequel ont été réalisés des impacts. 

## Fiche de route :

Voici la découpe des différents objectifs.

### Méthodes de localisation :

- [x] Trilatération de Kundu et al.
  
- [ ] Réseau de neurones 

### Méthodes de calcul des différences des temps d'arrivée

- [x] Seuil naïf

- [x] Cross-corrélation

- [ ] Seuil à enveloppe

- [ ] Transformée en ondelettes

### Traitement des données

- [X] Moyenner les positions des impacteur et/ou les positions statiques.

- [x] Calcul de valeurs statistiques (Moyenne, Médiane, Écart-type)

- [x] Sauvegarde des données dans un fichier pour traiter les différences entre méthodes

- [x] Affichage des différences en médiane et écart-type des différentes méthodes

- [ ] Utilisation ergonomique de l'affichage

- [X] Vérification de la similitude entre données par un test statistique (Shapiro, one way ANOVA, TukeyHSD, Kruskal Wallis, Dunn)


### Intégration

- [X] Intégrer les différentes méthodes au sein d'un seul code cohérent et simple d'usage

- [ ] Mise en place d'un script bash pour automatiser les lancements de différents paramètres


## Paramètres de réglage implémentés

- typeLocalisation
  - [X]  Trilateration
- typeTdA
  - [X] SeuilNaif
  - [X] CrossCorrelation
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
  - [X] SurtapisSautStage
  - [X] SurtapisImpactStage
  - [X] SurtapisToutStage
  - [X] SurtapisSautMiniProjet
  - [X] SurtapisImpactMiniProjet
  - [X] SurtapisStatiqueImpactMiniProjet
  - [X] SurtapisToutMiniProjet
  - [X] TapisSautStage
  - [X] TapisImpactStage
  - [X] TapisToutStage
  - [X] TapisSautMiniProjet
  - [X] TapisStatiqueSautMiniProjet
  - [X] TapisImpactMiniProjet
  - [X] TapisStatiqueImpactMiniProjet
  - [X] TapisToutMiniProjet


## Résultats

### Type optimisation

Les différents type d'optimisation et le test de Kruskal Wallis montre que les résultats sont similaires avec un seuil de confiance de 5%. Nous resterons donc sur la méthode **Default**.

### Méthode de TdA

Les trois méthodes, testées avec Kruskal Wallis sur TapisSautStage, ont une différence significative pour chaque méthode entre elles, et la meilleure est la crossCorrelation (voir le graphique).

Ce document est voué à être modifié au fur et à mesure.
