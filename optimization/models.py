from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def PCA_wrapper(trial):
    n_components = trial.suggest_int('n_components',1,m, log = True)
    return PCA(n_components= n_components, n_jobs=-1) 

def KPCA_wrapper(trial):
    n_components = trial.suggest_int('n_components',1,m, log = True)
    kernel = trial.suggest_categorical('kernal',['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
    return KernelPCA(n_components= n_components, kernel = kernel, n_jobs=-1)

def LDA_wrapper(trial):
    solver = trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen'])
    return LinearDiscriminantAnalysis(solver = solver)

def SVD_wrapper(trial):
    n_components = trial.suggest_int('n_components',1,m, log = True)
    return TruncatedSVD(n_components = n_components)

def Lasso_wrapper(trial):
    alpha = trial.suggest_int('alpha',.001,1000, log = True)
    return Lasso(alpha = alpha)

def LR_wrapper(trial):
    penalty = trial.suggest_categorical('penalty',[None,'l2','l1','elasticnet'])
    return LogisticRegression(penalty=penalty)

def NB_wrapper(trial):
    return GaussianNB()

def DT_wrapper(trial):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    return DecisionTreeClassifier(criterion = criterion)

def random_forest_wrapper(trial):
    n_estimators = trial.suggest_int('n_estimators',1, 1000, log = True)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    return RandomForestClassifier(n_estimators=n_estimators,criterion=criterion)

def kNN_wrapper(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, n_con, log = True)
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def kernal_SVM_wrapper(trial):
    kernal = trial.suggest_categorical('kernal', ['linear', 'poly', 'rbf', 'sigmoid'])
    return svm.SVC(kernal=kernal)

def xg_boost_wrapper(trial):
    booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
    max_depth = trial.suggest_int('max_depth',.001,1000,log = True)
    return xgb.XGBClassifier(booster = booster, max_depth=max_depth)

def NN_wrapper(trial):
    solver = trial.suggest_categorical(solver,'lbfgs', 'sgd', 'adam')
    alpha = trial.suggest_float('alpha',.001,1000,log = True)
    hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes',.001,1000,log = True)
    return MLPClassifier(solver = solver, alpha = alpha, hidden_layer_sizes = hidden_layer_sizes)

# Maps dimensionality reduction technique to its wrapper
dim_reduction_mappings = {
        None: None,
        'PCA': PCA_wrapper,
        'KPCA': KPCA_wrapper,
        'LDA': LDA_wrapper,
        'SVD': SVD_wrapper,
        'LASSO': LDA_wrapper
}

# Maps classifier to its wrapper
clf_mappings = {
        'LR': LR_wrapper,
        'NB': NB_wrapper,
        'DT': DT_wrapper,
        'random_forest': random_forest_wrapper,
        'kNN': kNN_wrapper,
        'SVM': kernal_SVM_wrapper,
        'XGBoost': xg_boost_wrapper,
        'NN': NN_wrapper
}