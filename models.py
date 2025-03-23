# Get EEG signal
# signal = pyedflib.highlevel.read_edf(str(cassette_path / psg_filename))[0][0]
#
# Sleep waves
# Delta: 0-3.5 Hz
# Theta: 4-7.5 Hz
# Alpha: 8-13 Hz
# Beta: 14-30 Hz
# Gamma 30-100 Hz

from sklearn.preprocessing import StandardScaler as sc
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


#PCA/SVD Model selection scoring method, cite https://stackoverflow.com/questions/53556359/selecting-kernel-and-hyperparameters-for-kernel-pca-reduction
def my_scorer(estimator, X, y=None):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X, X_preimage)

#PCA Model selection
def model_select_pca(x):
    sc_X = sc()
    x[x.columns] = sc_X.fit_transform(x[x.columns])
    param_grid = [{"gamma": np.linspace(0.01, 1, 5),"kernel": ["rbf", "sigmoid", "linear", "poly"]}]
    kpca=KernelPCA(fit_inverse_transform=True, n_jobs=-1) 
    grid_search = GridSearchCV(kpca, param_grid, cv=5, scoring=my_scorer)
    grid_search.fit(x)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    return x

#SVD
def SVD(x):
    param_grid = [{"n_components": np.linspace(1, 10, 10),}]
    trun_svd =  TruncatedSVD()
    grid_search = GridSearchCV(trun_svd, param_grid, scoring=my_scorer)
    x_new = grid_search.fit(x)
    return x_new

#LDA

#LASSO

#Random Forest