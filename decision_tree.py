import numpy as np

class Noeud():
    def __init__(self,X,y,depth,classes,critere="gini",feuille=False):
        self.feuille = feuille
        self.X = X
        self.y = y
        self.depth = depth
        self.classes = classes
        self.n1 = None
        self.n2 = None
        self.critere = critere
        self.featureIndex = None
        self.tv = None
        self.isClassification = critere in ['gini','entropy']

    def setFeuille(self):
        self.feuille = True
    def setTest(self,featureIndex,tv,quali):
        self.featureIndex = featureIndex
        self.tv = tv
        self.isQualitatif = quali
        
    def setFeuilleGauche(self,n):
        self.n1 = n
    def setFeuilleDroite(self,n):
        self.n2 = n

    def isFeuille(self):
        return self.feuille or self.getPurete() == 0
    def getPlusFrequent(self):
        from collections import Counter
        return Counter(self.y).most_common(1)[0][0]

    # la moyenne des valeurs da la cible
    def getMoy(self):
        return self.y.mean()

    # predire la valeur de la cible de donnée "pre"
    def getPrediction(self,pre):
        # si le noeud est une feuille on donne la valeur de getPlusFrequent ou la mpoyenne (getMoy)
        if self.n1 is None or self.n2 is None:
            if self.isClassification:
                return self.getPlusFrequent()
            else:
                return self.getMoy()
        # sinon, on decide si le noeud fils qui va donner la prediction ou le droit
        f = pre[self.featureIndex]
        if self.isQualitatif:
            return self.n1.getPrediction(pre) if f == self.tv else self.n2.getPrediction(pre)
        else:
            return self.n1.getPrediction(pre) if f <= self.tv else self.n2.getPrediction(pre)
    # la pureté d'un noeud selon la critère
    def getPurete(self):
        fn = {
            'gini' : self.gini,
            'entropy' : self.entropy,
            'squared_error' : self.squared_error
        }
        return fn[self.critere]()
    # critère "squared_error" de type regression
    def squared_error(self):
        return ((self.y - self.y.mean())**2).sum() / self.y.size
    # critère "entropy" de type classification
    def entropy(self):
        occu = self.getOcuu()
        p = []
        for i in occu:
            if i!=0:
                p.append(i)
        p = np.array(p)
        p = (p / p.sum())
        a = p * np.log(p)
        return -1 * a.sum()
    # critère "gini" de type classification
    def gini(self):
        occu = self.getOcuu()
        p = (occu / occu.sum())
        a = p * (1 - p)
        return a.sum()

    def getSum(self):
        return self.X.shape[0]
    # les occurences des classes de la cible dans ce noeud
    def getOcuu(self):
        a = [int(i) for i in self.y]
        a = np.bincount(a)
        occu = [(a[i] if i < a.size else 0) for i in self.classes]
        return np.array(occu)
    # afficher des informations de ce noeud dans une manière hiérarchique
    def print(self):
        a = ""
        if self.featureIndex is not None and self.tv is not None:
            a = "==" if self.isQualitatif else "<="
            a = "X[%d] %s %.2f" % (self.featureIndex,a,self.tv)
        print(self.depth * "\t",
              self.getOcuu() if self.isClassification else "",
              "samples:",self.getSum(),
              a,
              self.critere+":",self.getPurete())
        if self.n1 is not None:
            self.n1.print()
        if self.n2 is not None:
            self.n2.print()
            
    # clacule de cost on fonction des deux noeud fils
    def getCost(self):
        sum1,p1 = self.n1.getSum(),self.n1.getPurete()
        sum2,p2 = self.n2.getSum(),self.n2.getPurete()
        res = (sum1 * p1 + sum2 * p2) / (sum1 + sum2)
        # print(sum1,p1,sum2,p2)
        return res


class DecisionTree():

    _class_criteres = ['gini','entropy']
    _regr_criteres = ['squared_error']
    _tous_criteres = ['gini','entropy','squared_error']

    def __init__(self,X,y,maxDepth=None,critere='gini'):
        
        self.maxDepth = maxDepth
        self.X = X
        self.y = y
        
        if critere not in DecisionTree._tous_criteres:
            raise Exception("critere invalid")
        self.isClassification = critere in DecisionTree._class_criteres
        self.critere = critere
        self.classes = np.unique(y) if self.isClassification else None
    # fonction d'entrainement
    def entrainer(self):
        self.racine = Noeud(self.X,self.y,0,classes=self.classes,critere=self.critere)
        return self._entrainer(self.racine)
    def _entrainer(self,noeud:Noeud):
        test_valeurs,droite,gauche = None,None,None
        
        puretes = []

        if (self.maxDepth is not None and noeud.depth >= self.maxDepth) or (noeud.isFeuille()):
            return
            
        pur = noeud.getPurete()
        for i in range(noeud.X.shape[1]):
            feature = noeud.X[:,i]
            isQualitatif = len(np.unique(feature)) <= 5
            if isQualitatif:
                test_valeurs = np.unique(feature)
            else:
                sorted = np.sort(feature,axis=0)
                test_valeurs = (sorted[1:] + sorted[:-1])/2
            for tv in test_valeurs:
                if self.split(noeud,feature,tv,isQualitatif) != False:
                    c = noeud.getCost()
                    if self.isClassification:
                        puretes.append([c,i,tv])
                    else:
                        if c < pur:
                            puretes.append([c,i,tv])
        noeud.setFeuilleGauche(None)
        noeud.setFeuilleDroite(None)
        if len(puretes) == 0:
            return
        puretes = np.array(puretes)
        min_ = puretes[:,0].argmin()
        c,i,tv = puretes[min_]
        i = int(i)
        
        feature = noeud.X[:,i]
        isQualitatif = len(np.unique(feature)) <= 5
        self.split(noeud,feature,tv,isQualitatif)
        noeud.setTest(i,tv,isQualitatif)
        self._entrainer(noeud.n1)
        self._entrainer(noeud.n2)
    # affichage d'arbre
    def print(self):
        self.racine.print()
    # predire la classe d'une ou plusieurs données
    def predict(self,pre):
        pre = np.array(pre)
        if pre.ndim == 2:
            return np.array([self.racine.getPrediction(i) for i in pre])
        elif pre.ndim == 1:
            return self.racine.getPrediction(pre)
    # diviser le jeu de données en deux parties en fonction de feature et tv 
    def split(self,noeud,feature,tv,isQualitatif):
        id_gauche = None
        id_droite = None
        if isQualitatif:
            gauche = noeud.X[feature == tv]
            droite = noeud.X[feature != tv]
            id_gauche = np.where(feature == tv)
            id_droite = np.where(feature != tv)
        else:
            gauche = noeud.X[feature <= tv]
            droite = noeud.X[feature > tv]
            id_gauche = np.where(feature <= tv)
            id_droite = np.where(feature > tv)
        id_gauche,id_droite = id_gauche[0],id_droite[0]
        if id_gauche.size == 0 or id_droite.size == 0:
            return False
        noeud.setFeuilleGauche(Noeud(gauche,noeud.y[id_gauche],noeud.depth+1,classes=self.classes,critere=noeud.critere))
        noeud.setFeuilleDroite(Noeud(droite,noeud.y[id_droite],noeud.depth+1,classes=self.classes,critere=noeud.critere))