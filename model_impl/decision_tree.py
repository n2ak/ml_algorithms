import numpy as np


def getOcuu(y, classes):
    a = [int(i) for i in y]
    a = np.bincount(a)
    occu = [(a[i] if i < a.size else 0) for i in classes]
    return np.array(occu)


def entropy(y, classes):
    occu = getOcuu(y, classes)
    p = []
    for i in occu:
        if i != 0:
            p.append(i)
    p = np.array(p)
    p = (p / p.sum())
    a = p * np.log(p)
    return -1 * a.sum()


def gini(y, classes):
    occu = getOcuu(y, classes)
    p = (occu / occu.sum())
    a = p * (1 - p)
    return a.sum()


def squared_error(y):
    return ((y - y.mean())**2).sum() / y.size


class Node():
    def __init__(self, X, y, depth, criterion="gini", max_depth=10):
        self.X = X
        self.y = y
        self.depth = depth
        self.left: Node = None
        self.right: Node = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.is_class = criterion in ['gini', 'entropy']
        self.feature_value = None
        if self.is_class:
            self.classes = np.unique(y)

        self.result = None

    def set_left(self, n):
        self.left = n

    def set_right(self, n):
        self.right = n

    fn = {
        'gini': gini,
        'entropy': entropy,
        'squared_error': squared_error
    }

    def get_purity(self):
        criteria = self.fn[self.criterion]
        if self.is_class:
            return criteria(self.y, self.classes)
        return criteria(self.y)

    def isleaf(self):
        return (self.left is None and self.right is None) or self.get_purity() == 0

    def get_probabilities(self, all_classes):
        ret = np.zeros(len(all_classes))
        for i, c in enumerate(all_classes):
            indices = np.where(self.y == c)[0].shape[0]
            ret[i] = indices
        ret /= self.y.size
        return ret

    def get_mean(self):
        return self.y.mean()

    def get_result(self, all_classes):
        if self.result is None:
            self.result = self.get_probabilities(
                all_classes) if self.is_class else self.get_mean()
        return self.result

    def predict(self, pre: np.ndarray, all_classes=None):
        assert pre.ndim == 1
        # if self.left is None or self.right is None:
        if self.isleaf():
            return self.get_result(all_classes)
        feature, value, is_discrete = self.feature_value
        f = pre[feature]
        go_left = f == value if is_discrete else f <= value
        next_node = self.left if go_left else self.right
        return next_node.predict(pre, all_classes)

    def number_of_samples(self):
        return self.y.size

    def print(self):
        a = ""
        if self.feature_value is not None:
            feature_index, value, is_discrete = self.feature_value
            a = "==" if is_discrete else "<="
            a = "X[%d] %s %.2f" % (feature_index, a, value)
        print(self.depth * "\t",
              getOcuu(self.y, self.classes) if self.is_class else "",
              "samples:", self.number_of_samples(),
              a,
              self.criterion+":", self.get_purity())
        if self.left is not None:
            self.left.print()
        if self.right is not None:
            self.right.print()

    def getCost(self, left: "Node", right: "Node"):
        sum1, p1 = left.number_of_samples(), left.get_purity()
        sum2, p2 = right.number_of_samples(), right.get_purity()
        res = (sum1 * p1 + sum2 * p2) / (sum1 + sum2)
        return res

    def fit(self):
        test_values, right, left = None, None, None
        node = self
        puritys = []
        if (self.max_depth is not None and node.depth >= self.max_depth):
            return
        pur = node.get_purity()
        nf = node.X.shape[1]
        min_nodes = pur, None, None, -1, None
        for i in range(nf):
            feature = node.X[:, i]
            unique_values = np.unique(feature)
            is_discrete = len(unique_values) <= 5
            if is_discrete:
                test_values = unique_values
            else:
                sorted = np.sort(feature, axis=0)
                test_values = (sorted[1:] + sorted[:-1])/2

            for tv in test_values:
                left, right = self.split(feature, tv, is_discrete)
                if left.sum() != 0 and right.sum() != 0:
                    left_node, right_node = self.new_node(
                        left), self.new_node(right)
                    c = self.getCost(left_node, right_node)
                    if False:  # self.isClassification:
                        puritys.append([c, i, tv])
                    else:
                        if c < min_nodes[0]:
                            min_nodes = c, left_node, right_node, i, tv, is_discrete
        if min_nodes[1] is None:
            return
        _, left, right, f, v, is_discrete = min_nodes
        self.feature_value = f, v, is_discrete
        self.set_left(left)
        self.set_right(right)
        self.left.fit()
        self.right.fit()
        return self

    def new_node(self, indices):
        return Node(self.X[indices], self.y[indices], self.depth+1, criterion=self.criterion)

    def split(self, feature, tv, is_discrete):
        left_idx = (feature == tv) if is_discrete else (feature <= tv)
        right_idx = np.invert(left_idx)
        return left_idx, right_idx


class DecisionTree():

    _class_criterions = ['gini', 'entropy']

    def __init__(self, criterion, max_depth=None):
        self.max_depth = max_depth
        if criterion not in self._criterions:
            raise Exception(
                f"invalid criterion '{criterion}', possible criterions: {self._criterions}")
        self.criterion = criterion

    def fit(self, X, y):
        self.root = Node(
            X, y,
            0,
            criterion=self.criterion,
            max_depth=self.max_depth,
        ).fit()
        self.all_classes = np.unique(y)
        return self

    def print(self):
        self.root.print()

    def predict_proba(self, pre):
        pre = np.atleast_2d(pre)
        return np.array([self.root.predict(i, self.all_classes) for i in np.atleast_2d(pre)])

    def predict(self, pre):
        if self.criterion in self._class_criterions:
            pred = self.predict_proba(pre).argmax(-1)
            # we go backwards cuz `all_classes` is sorted
            for i in range(len(self.all_classes)-1, -1, -1):
                c = self.all_classes[i]
                pred[pred == i] = c
            return pred

        return np.array([self.root.predict(i) for i in np.atleast_2d(pre)])

    def get_params(self, *args, **kwargs):
        # print(kwargs)
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth
        }
