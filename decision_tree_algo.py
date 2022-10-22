

def entr(noeud,depth):
    if depth >= max_depth:
        return
    for feature in features:
        if isQualitatif(feature):
            valeurs = numpy.unique(target) 
        else:
            valeurs = (target[1:] + target[:-1]) / 2 
        for valeur in values:
            costs = []
            gauche,droite = split(dataset,feature,valeur)
            setNoeudFils(noeud,gauche,droite)
            if isClassification:
                cost = getCost(critere = "gini")# ou entropy
            else:
                cost = getCost(critere = "squared_error")
            if cost != 0:
                costs.append(cost)
    min_cost = min(costs)
    if min_cost != None:
        setNoeudFils(noeud,gauche,droite)
        entr(noeud.gauche,depth + 1)
        entr(noeud.droite,depth + 1)

def split(dataset,feature,valeur):
    if isQualitatif(feature):
        gauche = dataset[dataset == valeur]
        droite = dataset[dataset != valeur]
    else:
        gauche = dataset[dataset <= valeur]
        droite = dataset[dataset > valeur]
    return gauche,droite



def main():
    racine = Noued(dataset,depth=0)
    entr(racine)