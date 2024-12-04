

# Tous les parametres implementes, mais tous les tester necessiterait, en supposant 15 secondes par passage (très optimiste), 47 h non-stop. Donc on va degrossir petit a petit.

TYPE_LOCALISATION=("Trilateration")
TYPE_TDA=("SeuilNaif" "CrossCorrelation" "SeuilEnveloppe")
TYPE_OPTIMISATION=("Default" "Nelder-Mead" "Powell" "CG" "BFGS" "Newton-CG" "L-BFGS-B" "TNC" "COBYLA" "SLSQP" "trust-constr" "dogleg" "trust-ncg" "trust-exact" "trust-krylov")
VALEUR_SEUIL=(0.5 1.0 2.0 4.0 7.0 10.0)
TRAITEMENT_ACCELEROMETRE=("AxeZ" "Norme")
DATA_SET=("SurtapisSautStage" "SurtapisImpactStage" "SurtapisToutStage" "SurtapisSautMiniProj" "SurtapisImpactMiniProj" "SurtapisToutMiniProj" "SurtapisTout" "TapisSautStage" "TapisImpactStage" "TapisToutStage" "TapisSautMiniProj" "TapisImpactMiniProj" "TapisToutMiniProj" "TapisTout" "TapisStatiqueSautMiniProj" "TapisStatiqueImpactMiniProj" "SurtapisStatiqueImpactMiniProj")

# On choisit ceux qu'on veut tester pour cette run


TYPE_TDA="SeuilNaif"
VALEUR_SEUIL=4.0
TRAITEMENT_ACCELEROMETRE="Norme"
DATA_SET="TapisSautStage"

for t in ${TYPE_OPTIMISATION[@]}; do
   mkdir Images/${TYPE_LOCALISATION}_${TYPE_TDA}_${t}_${VALEUR_SEUIL}_${TRAITEMENT_ACCELEROMETRE}_${DATA_SET}  
   python3 main.py ${TYPE_LOCALISATION} ${TYPE_TDA} ${t} ${VALEUR_SEUIL} ${TRAITEMENT_ACCELEROMETRE} ${DATA_SET}
done


