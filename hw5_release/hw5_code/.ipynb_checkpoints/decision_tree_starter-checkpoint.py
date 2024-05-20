# You may want to install "gprof2dot"
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate
import sklearn.tree
from tqdm import tqdm

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


# Vectorized function for hashing for np efficiency
def w(x):
    return np.int(hash(x)) % 1000


h = np.vectorize(w)

spam_features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]

titanic_features = scipy.io.loadmat("datasets/titanic/prep_titanic.mat")["feature_labels"][0]


class DecisionTree:
    
    class DTNode:
        def __init__(self, 
                     feat = None, 
                     thresh = None, 
                     left = None, 
                     right = None, 
                     label = None):
            self.feat = feat
            self.thresh = thresh
            self.left = left
            self.right = right
            self.label = label
        
        def __str__(self, level=0):
            if self.feat != None or self.thresh != None:
                ret = "\t"*level+repr("Feature x" + str(self.feat) + " > " + str(self.thresh))+"\n"
            else:
                ret = "\t"*level+repr("Leaf Label: " + str(self.label))+"\n"
            if self.left:
                ret += self.left.__str__(level+1)
            if self.right:
                ret += self.right.__str__(level+1)
            return ret
        
        def strtitanic(self, level=0):
            if self.feat != None or self.thresh != None:
                ret = "\t"*level+repr(titanic_features[self.feat][0] + " > " + str(self.thresh))+"\n"
            else:
                ret = "\t"*level+repr("Leaf Label: " + str(self.label))+"\n"
            if self.left:
                ret += self.left.strtitanic(level+1)
            if self.right:
                ret += self.right.strtitanic(level+1)
            return ret

        def __repr__(self):
            return '<tree node representation>'
    
    def __init__(self, max_depth=3, feature_labels=None):
        # TODO implement __init__ function
        self.max_depth = max_depth
        self.features = feature_labels
        self.tree = None
    
    def create_node(self, X, y, depth = 0):
        ent = self.entropy(y)
        if ent == 0 or depth == self.max_depth or len(y) == 1:
            l = max(set(y), key = (lambda x: np.count_nonzero(y == x)))
            assert l != None
            return self.DTNode(label = l)
        else:
            val, feat, thresh, ridx = self.best_split2(X, y)
            if val < 0.001:
                l = max(set(y), key = (lambda x: np.count_nonzero(y == x)))
                assert l != None
                return self.DTNode(label = l)
#            print(len(ridx), val, thresh, max(X[:, feat]), min(X[:, feat]))
            X_l = np.delete(X, ridx, axis = 0)
            y_l = np.delete(y, ridx)
            assert len(y_l) != 0
            X_r = X[ridx]
            y_r = y[ridx]
            assert len(y_r) != 0, str(ent) + " " + str(val)
            return self.DTNode(feat = feat, 
                               thresh = thresh, 
                               left = self.create_node(X_l, y_l, depth = depth + 1), 
                               right = self.create_node(X_r, y_r, depth = depth + 1))
    
    def best_split(self, X, y):
        best_infog_val = 0
        best_infog_thresh = 0
        best_infog_feat = 0
        for i in range(X.shape[1]): # looping through features
            mx = np.max(X[:, i])
            mn = np.min(X[:, i])
            check = np.linspace(mn, mx, 20, endpoint = False)[1:]
            for t in check:
                ig = self.information_gain(X, y, i, t)
                if ig > best_infog_val:
                    best_infog_val = ig
                    best_infog_thresh = t
                    best_infog_feat = i
        
        best_ridx = self.idx_split(X, y, best_infog_feat, best_infog_thresh)
        
        
        return best_infog_val, best_infog_feat, best_infog_thresh, best_ridx
    
    def best_split2(self, X, y):
        best_infog_val = 0
        best_infog_thresh = 0
        best_infog_feat = 0
        for i in range(X.shape[1]): # looping through features
            check = X[:, i][1:] - 1e-5
            if len(check) > 20:
                mx = np.max(X[:, i])
                mn = np.min(X[:, i])
                check = np.linspace(mn, mx, 20, endpoint = False)[1:]
            for t in check:
                ig = self.information_gain(X, y, i, t)
                if ig > best_infog_val:
                    best_infog_val = ig
                    best_infog_thresh = t
                    best_infog_feat = i
        
        best_ridx = self.idx_split(X, y, best_infog_feat, best_infog_thresh)
        
        
        return best_infog_val, best_infog_feat, best_infog_thresh, best_ridx
    

    def information_gain(self, X, y, feat_idx, thresh):
        ridx = self.idx_split(X, y, feat_idx, thresh)
        y_l = np.delete(y, ridx)
        y_r = y[ridx]
        
        tot = len(y)
        num_r = len(y_r)
        num_l = tot - num_r
        
        ent_b = self.entropy(y)
        ent_l = self.entropy(y_l)
        ent_r = self.entropy(y_r)
        ent_avg = (num_l * ent_l + num_r * ent_r) / tot
        
        return ent_b - ent_avg

    def entropy(self, labels):
        freq = Counter(labels)
        clas = np.fromiter(freq.keys(), dtype = int)
        num = np.fromiter(freq.values(), dtype = int)
        pc = num / np.sum(num)
        arr = pc * np.log2(pc)
        return -np.sum(arr)

    def idx_split(self, X, y, idx, thresh):
        """ Returns the indices which are in right node."""
        ftlist = X[:, idx]
#        print(min(ftlist), max(ftlist))
        ridx = np.where(ftlist > thresh)[0]
        return ridx
    

    def fit(self, X, y, depth = 0):
        self.tree = self.create_node(X, y, depth = depth)
        return self.tree
    
    def get_pred(self, pt, t = None):
        if t == None:
            t = self.tree
        if t.label == None:
            if pt[t.feat] > t.thresh:
                return self.get_pred(pt, t.right)
            else:
                return self.get_pred(pt, t.left)
        else:
            return t.label
    
    def trace_path_spam(self, pt, t = None):
        if t == None:
            t = self.tree
        if t.label == None:
            if pt[t.feat] > t.thresh:
                print(spam_features[t.feat] + " > " + str(t.thresh))
                return self.trace_path_spam(pt, t.right)
            else:
                print(spam_features[t.feat] + " <= " + str(t.thresh))
                return self.trace_path_spam(pt, t.left)
        else:
            d = {0: "(SPAM)", 1: "(HAM)"}
            print("Classified as: " + str(t.label) + " " + d[t.label])
            return t.label
            

    def predict(self, X):
        res = np.apply_along_axis(self.get_pred, 1, X)
        return res
    
    def accuracy(self, pred, actual):
        return np.sum(actual == pred) / len(pred)
    

class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # ROTOTODO
        pass

    def predict(self, X):
        # TODO implement function
        pass


class RandomForest(DecisionTree):
    def __init__(self, X, y, n_sub = 200, n_trees = 10, n_subf = 10, max_depth = 3):
        self.n_sub = n_sub
        self.n_subf = n_subf
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        idx = np.random.randint(0, len(y), size = n_sub)
        self.X = X[idx]
        self.y = y[idx]
        return
    
    def best_split2(self, X, y):
        best_infog_val = 0
        best_infog_thresh = 0
        best_infog_feat = 0
        feats = range(X.shape[1])
        remf = np.random.choice(feats, size = self.n_subf)
        for i in remf: # looping through features
            check = X[:, i][1:] - 1e-5
            if len(check) > 20:
                mx = np.max(X[:, i])
                mn = np.min(X[:, i])
                check = np.linspace(mn, mx, 20, endpoint = False)[1:]
 #               print(check, mn, mx)
            for t in check:
                ig = self.information_gain(X, y, i, t)
                if ig > best_infog_val:
                    best_infog_val = ig
                    best_infog_thresh = t
                    best_infog_feat = i
 #       print("check best")
        best_ridx = self.idx_split(X, y, best_infog_feat, best_infog_thresh)
        
        
        return best_infog_val, best_infog_feat, best_infog_thresh, best_ridx
    
    def fit(self, depth = 0):
        for i in tqdm(range(self.n_trees)):
            self.trees.append(self.create_node(self.X, self.y, depth = depth))
        return self.trees
    
    def predict(self, X):
        tot = 0
        for i in range(self.n_trees):
            if i == 0:
                tot = np.apply_along_axis(lambda x: self.get_pred(i, x), 1, X)
            else:
                res = np.apply_along_axis(lambda x: self.get_pred(i, x), 1, X)
                tot = tot + res
        tot = tot / self.n_trees
        return tot
    
    def get_pred(self, i, pt, t = None):
        if t == None:
            t = self.trees[i]
        if t.label == None:
            if pt[t.feat] > t.thresh:
                return self.get_pred(i, pt, t.right)
            else:
                return self.get_pred(i, pt, t.left)
        else:
            return t.label

# You do not have to implement the following boost part, though it might help with Kaggle.
class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO implement function
        return self

    def predict(self, X):
        # TODO implement function
        pass



def evaluate(clf):
    print("Cross validation:")
    cv_results = cross_validate(clf, X, y, cv=5, return_train_score=True)
    train_results = cv_results['train_score']
    test_results = cv_results['test_score']
    avg_train_accuracy = sum(train_results) / len(train_results)
    avg_test_accuracy = sum(test_results) / len(test_results)

    print('averaged train accuracy:', avg_train_accuracy)
    print('averaged validation accuracy:', avg_test_accuracy)
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)

    return avg_train_accuracy, avg_test_accuracy


if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data       
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]
        
        # TODO: preprocess titanic dataset
        # Notes: 
        # 1. Some data points are missing their labels
        # 2. Some features are not numerical but categorical
        # 3. Some values are missing for some features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = './datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print('==================================================')
    print("\n\nSimplified decision tree")
    dt = DecisionTree(max_depth=3, feature_labels=features)
    dt.fit(X, y)
    print("Predictions", dt.predict(Z)[:100])
    print("Tree structure", dt.__repr__())

    # TODO implement and evaluate remaining parts
