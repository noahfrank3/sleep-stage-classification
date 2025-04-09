from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

class DimReductionWrapper():
    def __init__(self, trial, n, m, n_y):
        self.trial = trial
        self.max_size = min(n, m)
        self.max_size_y = min(self.max_size, n_y)

    def get_dim_reduction(self):
        dim_reduction_mappings = {
            None: None,
            'KPCA': self.PCA_wrapper,
            'LDA': self.LDA_wrapper,
            'SVD': self.SVD_wrapper,
            'LASSO': self.LDA_wrapper
        }

        dim_reduction = self.trial.suggest_categorical('dim_reduction', list(dim_reduction_mappings.keys()))

        if dim_reduction is None:
            dim_reduction = 'passthrough'
        else:
            dim_reduction = dim_reduction_mappings[dim_reduction]()

        return dim_reduction

    def PCA_wrapper(self):
        n_components = self.trial.suggest_int('n_components_PCA', 1, self.max_size, log=True)
        kernel = self.trial.suggest_categorical('kernal_PCA', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])

        if kernel in ['linear', 'cosine']:
            gamma = None
        else:
            gamma = self.trial.suggest_float('gamma_PCA', 0.001, 1000, log=True)

        if kernel == 'poly':
            degree = self.trial.suggest_int('degree_PCA', 2, 10)
        else:
            degree = 3

        if kernel in ['poly', 'sigmoid']:
            coef0 = self.trial.suggest_float('coef0_PCA', -2, 2)
        else:
            coef0 = 1

        return KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, n_jobs=-1)

    def LDA_wrapper(self):
        n_components = self.trial.suggest_int('n_components_LDA', 1, self.max_size_y, log=True)
        return LinearDiscriminantAnalysis(n_components=n_components)

    def SVD_wrapper(self):
        n_components = self.trial.suggest_int('n_components_SVD', 1, self.max_size, log=True)
        return TruncatedSVD(n_components=n_components)

    def Lasso_wrapper(self):
        alpha = self.trial.suggest_float('alpha_Lasso', 0.001 , 1000, log=True)
        return Lasso(alpha=alpha)

class CLFWrapper():
    def __init__(self, trial, n, m):
        self.trial = trial
        self.max_size = min(n, m)

    def get_clf(self):
        clf_mappings = {
            'QDA': self.QDA_wrapper,
            'LR': self.LR_wrapper,
            'NB': self.NB_wrapper,
            'DT': self.DT_wrapper,
            'RF': self.RF_wrapper,
            'kNN': self.kNN_wrapper,
            'SVM': self.SVM_wrapper,
            'XGB': self.XGB_wrapper,
            'NN': self.NN_wrapper
        }

        clf = self.trial.suggest_categorical('clf', list(clf_mappings.keys()))
        return clf_mappings[clf]()

    def QDA_wrapper(self):
        return QuadraticDiscriminantAnalysis()

    def LR_wrapper(self):
        penalty = self.trial.suggest_categorical('penalty_LR', [None, 'l2', 'l1', 'elasticnet'])
        
        if penalty is None:
            C = 1
        else:
            C = self.trial.suggest_float('C_LR', 0.001, 1000, log=True)
        
        if penalty == 'elasticnet':
            l1_ratio = self.trial.suggest_float('l1_ratio_LR', 0, 1)
        else:
            l1_ratio = None

        return LogisticRegression(solver='saga', penalty=penalty, C=C, l1_ratio=l1_ratio, n_jobs=-1)

    def NB_wrapper(self):
        return GaussianNB()

    def DT_wrapper(self):
        max_depth = self.trial.suggest_int('max_depth_DT', 1, self.max_size, log=True)
        max_leaf_nodes = self.trial.suggest_int('max_leaf_nodes_DT', 1, self.max_size, log=True)
        return DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)

    def RF_wrapper(self):
        n_estimators = self.trial.suggest_int('n_estimators_RF', 1, 1000, log=True)
        max_depth = self.trial.suggest_int('max_depth_RF', 1, self.max_size, log=True)
        max_leaf_nodes = self.trial.suggest_int('max_leaf_nodes_RF', 1, self.max_size, log=True)
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, n_jobs=-1)

    def kNN_wrapper(self):
        n_neighbors = self.trial.suggest_int('n_neighbors_kNN', 1, self.max_size, log=True)
        weights = self.trial.suggest_categorical('weights_kNN', ['uniform', 'distance'])
        return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=-1)

    def SVM_wrapper(self):
        C = self.trial.suggest_float('C', 0.001, 1000, log=True)
        kernel = self.trial.suggest_categorical('kernal_SVM', ['linear', 'poly', 'rbf', 'sigmoid'])

        if kernel == 'cosine':
            gamma = 'scale'
        else:
            gamma = self.trial.suggest_float('gamma_SVM', 0.001, 1000, log=True)

        if kernel == 'poly':
            degree = self.trial.suggest_int('degree_SVM', 2, 10)
        else:
            degree = 3

        if kernel in ['poly', 'sigmoid']:
            coef0 = self.trial.suggest_float('coef0_SVM', -2, 2)
        else:
            coef0 = 0

        return svm.SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

    def XGB_wrapper(self):
        booster = self.trial.suggest_categorical('booster_XGB', ['gbtree', 'gblinear', 'dart'])
        max_depth = self.trial.suggest_int('max_depth_XGB', 1, 1000, log=True)
        return xgb.XGBClassifier(booster=booster, max_depth=max_depth, n_jobs=-1)

    def NN_wrapper(self):
        n_hidden_layers = self.trial.suggest_int('n_hidden_layers_NN', 1, 10)
        hidden_layers_size = []
        for idx in range(n_hidden_layers):
            hidden_layers_size.append(self.trial.suggest_int(f'hidden_layer_{idx}_size_NN', 1, 250))
        return MLPClassifier(hidden_layer_sizes=hidden_layers_size)
