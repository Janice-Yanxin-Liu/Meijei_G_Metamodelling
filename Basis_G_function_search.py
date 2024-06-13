import os
os.chdir('/gpfs/home/s346121/Meta_XGBoost')
import sys
sys.path.append('/gpfs/home/s346121/Meta_XGBoost')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import scipy as sc

import itertools
import copy

from mpmath import *
from sympy import *

from sympy.utilities.autowrap import ufuncify

from pysymbolic.models.special_functions import *

from tqdm import tqdm, trange, tqdm_notebook, tnrange

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge
from sympy import Integral, Symbol
from sympy.abc import x, y

from scipy.optimize import minimize

from pysymbolic.models.special_functions import MeijerG
from pysymbolic.utilities.performance import compute_Rsquared

import csv

df = pd.read_excel('df_xgboost_new.xlsx', index_col = 0, header = 0)

X = df[['Lipid', 'Lignin', 'Carbohydrate']]
k = df['k']

##-------------------------------------- DEFINE FUNCTIONS ------------------------------------- ##
def load_hyperparameter_config():

    hyperparameter_space = {
                        'hyper_1': (np.array([1.0, 1.0]), [1,0,0,1]),
                        'hyper_2': (np.array([1.0, 1.0, 1.0]), [1,0,0,2]),
                        'hyper_3': (np.array([1.0, 1.0, 1.0]), [2,0,0,2]),
                        'hyper_4': (np.array([1.0, 1.0, 1.0, 1.0]), [1,0,0,3]),
                        'hyper_5': (np.array([1.0, 1.0, 1.0, 1.0]), [2,0,0,3]),
                        'hyper_6': (np.array([1.0, 1.0, 1.0, 1.0]), [3,0,0,3]),
                        'hyper_7': (np.array([1.0, 1.0]), [0,1,1,0]),
                        'hyper_8': (np.array([1.0, 1.0, 1.0]), [0,1,1,1]),
                        'hyper_9': (np.array([3.3, 0.9, 1.0]), [1,0,1,1]),
                        'hyper_10': (np.array([1.0, 1.0, 1.0]), [1,1,1,1]),
                        'hyper_11': (np.array([1.0, 1.0, 1.0, 1.0]), [1,0,1,2]),
                        'hyper_12': (np.array([1.0, 1.0, 1.0, 1.0]), [2,0,1,2]),
                        'hyper_13': (np.array([1.0, 1.0, 1.0, 1.0]), [1,1,1,2]),
                        'hyper_14': (np.array([1.0, 1.0, 1.0, 1.0]), [2,1,1,2]),
                        'hyper_15': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [1,0,1,3]),
                        'hyper_16': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [2,0,1,3]),
                        'hyper_17': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [3,0,1,3]),
                        'hyper_18': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [1,1,1,3]),
                        'hyper_19': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [2,1,1,3]),
                        'hyper_20': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [3,1,1,3]),
                        'hyper_21': (np.array([3.1, 0.7, 1.0]), [0,1,2,0]),
                        'hyper_22': (np.array([1.0, 1.0, 1.0]), [0,2,2,0]),
                        'hyper_23': (np.array([3.0, 0.1, 0.3, 0.7, 1.0]), [0,1,3,1]),
                        'hyper_24': (np.array([2.7, 2.1, 0.3, 0.7, 1.0]), [0,2,3,1]),
                        'hyper_25': (np.array([1.0, 2.1, 0.3, 0.7, 1.0]), [0,3,3,1]),
                        'hyper_26': (np.array([2.7, 2.1, 0.3, 0.7, 1.0]), [1,0,3,1]),
                        'hyper_27': (np.array([3.1, 0.7, 0.3, 0.7, 1.0]), [1,1,3,1]),
                        'hyper_28': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [1,2,3,1]),
                        'hyper_29': (np.array([1.2, 1.1, 1.0, 1.0, 1.0]), [1,3,3,1]),
                        'hyper_30': (np.array([3.6, 0.9, -1.7, 1.0]), [0,1,2,1]),
                        'hyper_31': (np.array([1.0, 1.0, 1.0, 1.0]), [0,2,2,1]),
                        'hyper_32': (np.array([1.0, 1.0, 1.0, 1.0]), [1,0,2,1]),
                        'hyper_33': (np.array([1.0, 1.0, 1.0, 1.0]), [1,1,2,1]),
                        'hyper_34': (np.array([1.0, 1.0, 1.0, 1.0]), [1,2,2,1]),
                        'hyper_35': (np.array([0.0, 0.0, 0.0, 0.0, 1.0]), [0,1,2,2]),
                        'hyper_36': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [0,2,2,2]),
                        'hyper_37': (np.array([1.9, 0.0, 0.7, 1.7, 1.0]), [1,0,2,2]),
                        'hyper_38': (np.array([-0.9, 1.6, 0.3, -1.8, 1.0]), [1,1,2,2]),
                        'hyper_39': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [1,2,2,2]),
                        'hyper_40': (np.array([1.3, 1.0, 0.7, 0.8, 1.0]), [2,0,2,2]),
                        'hyper_41': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [2,1,2,2]),
                        'hyper_42': (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), [2,2,2,2]),
                        'hyper_43': (np.array([3.1, 0.4, 1.1, 1.0]), [0,1,3,0]),
                        'hyper_44': (np.array([3.1, 0.4, 1.1, 1.0]), [0,2,3,0]),
                        'hyper_45': (np.array([1.0, 1.0, 1.0, 1.0]), [0,3,3,0])
                        }

    return hyperparameter_space

def Optimize(Loss, theta_0):
    max_iterations = 10  

    for iteration in range(max_iterations):
        try:
            opt = minimize(Loss, theta_0, method='CG', options={'maxiter': 1, 'disp': True})
            Loss_ = opt.fun
            break
        
        except:
            print(f"Iteration {iteration}: {e}")
            theta_0 = theta_0 + np.random.randn()

    Loss_     = opt.fun
    
    return Loss_  

def symbolic_modeling(y, G_order, theta_0, X):

    def Loss(theta):
        
        try:        
            G     = MeijerG(theta=theta, order=G_order, evaluation_mode = 'eval')
            eval_result = G.evaluate(X)
        
        except ZeroDivisionError:
            G     = MeijerG(theta=theta, order=G_order, evaluation_mode = 'numpy')
            eval_result = G.evaluate(X)
        
        screen = ~ np.isnan(eval_result)

        loss_ = np.mean((y[screen] - eval_result[screen])**2)
        
        print("Loss:", loss_)
        
        return loss_
    
    Loss_ = Optimize(Loss, theta_0)

    return Loss_ 

def get_symbolic_model(Y_XG_pred, X):

    hyperparameter_space = load_hyperparameter_config() 

    losses_              = [] 

    for k in range(len(hyperparameter_space)):
        
        print(f"Optimising G-function {k+1}")

        try: 
            Loss_k = symbolic_modeling(Y_XG_pred, hyperparameter_space['hyper_'+str(k+1)][1], 
                                                  hyperparameter_space['hyper_'+str(k+1)][0], X)

            losses_.append(Loss_k)

        except:

            losses_.append('failure for G_order {k+1}')
    
    return losses_

## -------------- REPEAT SEARCHING MINIMUM LOSS ON EACH DIMENSION FOR 50 TIMES ------------- ##
X1_indices = []
X1_loss = []
X2_indices = []
X2_loss = []
X3_indices = []
X3_loss = []
X1X2_indices = []
X1X2_loss = []
X1X3_indices = []
X1X3_loss = []
X2X3_indices = []
X2X3_loss = []

for i in range (30):

    print('iteration {i}')
    X_train_k, X_test_k, k_train, k_test = train_test_split(X, k)
    xgb_k_red = xgb.XGBRegressor(early_stopping_rounds = 10, gamma = 1e-5, learning_rate = 0.2, max_depth = 4)
    xgb_k_red.fit(X_train_k, k_train, eval_set = [(X_test_k, k_test)], verbose = False)

    k_pred_for_meta = xgb_k_red.predict(X_train_k)

    X1 = np.array(X_train_k.iloc[:,0])
    X2 = np.array(X_train_k.iloc[:,1])
    X3 = np.array(X_train_k.iloc[:,2])

    feature_expander = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    Interaction_features = feature_expander.fit_transform(X_train_k)

    X1X2 = Interaction_features[:,3]
    X1X3 = Interaction_features[:,4]
    X2X3 = Interaction_features[:,5]

    # X1
    losses_X1 = get_symbolic_model(k_pred_for_meta, X1)
    losses_X1_array = np.array(losses_X1)

    sorted_indices_X1 = np.argsort(losses_X1_array).tolist()
    smallest_lossess_X1 = losses_X1_array[sorted_indices_X1].tolist()

    # X2
    losses_X2 = get_symbolic_model(k_pred_for_meta, X2)
    losses_X2_array = np.array(losses_X2)

    sorted_indices_X2 = np.argsort(losses_X2_array).tolist()
    smallest_lossess_X2 = losses_X2_array[sorted_indices_X2].tolist()

    # X3
    losses_X3 = get_symbolic_model(k_pred_for_meta, X3)
    losses_X3_array = np.array(losses_X3)

    sorted_indices_X3 = np.argsort(losses_X3_array).tolist()
    smallest_lossess_X3 = losses_X3_array[sorted_indices_X3].tolist()

    # X1X2
    losses_X1X2 = get_symbolic_model(k_pred_for_meta, X1X2)
    losses_X1X2_array = np.array(losses_X1X2)

    sorted_indices_X1X2 = np.argsort(losses_X1X2_array).tolist()
    smallest_lossess_X1X2 = losses_X1X2_array[sorted_indices_X1X2].tolist()

    # X1X3
    losses_X1X3 = get_symbolic_model(k_pred_for_meta, X1X3)
    losses_X1X3_array = np.array(losses_X1X3)

    sorted_indices_X1X3 = np.argsort(losses_X1X3_array).tolist()
    smallest_lossess_X1X3 = losses_X1X3_array[sorted_indices_X1X3].tolist()

    # X2X3
    losses_X2X3 = get_symbolic_model(k_pred_for_meta, X2X3)
    losses_X2X3_array = np.array(losses_X2X3)

    sorted_indices_X2X3 = np.argsort(losses_X2X3_array).tolist()
    smallest_lossess_X2X3 = losses_X2X3_array[sorted_indices_X2X3].tolist()

    X1_indices.append(sorted_indices_X1)
    X1_loss.append(smallest_lossess_X1)
    X2_indices.append(sorted_indices_X2)
    X2_loss.append(smallest_lossess_X2)
    X3_indices.append(sorted_indices_X3)
    X3_loss.append(smallest_lossess_X3)
    X1X2_indices.append(sorted_indices_X1X2)
    X1X2_loss.append(smallest_lossess_X1X2)
    X1X3_indices.append(sorted_indices_X1X3)
    X1X3_loss.append(smallest_lossess_X1X3)
    X2X3_indices.append(sorted_indices_X2X3)
    X2X3_loss.append(smallest_lossess_X2X3)

df_X1_indices = pd.DataFrame(np.array(X1_indices))
df_X1_loss = pd.DataFrame(np.array(X1_loss))
df_X2_indices = pd.DataFrame(np.array(X2_indices))
df_X2_loss = pd.DataFrame(np.array(X2_loss))
df_X3_indices = pd.DataFrame(np.array(X3_indices))
df_X3_loss = pd.DataFrame(np.array(X3_loss))
df_X1X2_indices = pd.DataFrame(np.array(X1X2_indices))
df_X1X2_loss = pd.DataFrame(np.array(X1X2_loss))
df_X1X3_indices = pd.DataFrame(np.array(X1X3_indices))
df_X1X3_loss = pd.DataFrame(np.array(X1X3_loss))
df_X2X3_indices = pd.DataFrame(np.array(X2X3_indices))
df_X2X3_loss = pd.DataFrame(np.array(X2X3_loss))

df_X1_indices.to_csv('df_X1_indices_k40.csv')
df_X1_loss.to_csv('df_X1_loss_k40.csv')
df_X2_indices.to_csv('df_X2_indices_k40.csv')
df_X2_loss.to_csv('df_X2_loss_k40.csv')
df_X3_indices.to_csv('df_X3_indices_k40.csv')
df_X3_loss.to_csv('df_X3_loss_k40.csv')
df_X1X2_indices.to_csv('df_X1X2_indices_k40.csv')
df_X1X2_loss.to_csv('df_X1X2_loss_k40.csv')
df_X1X3_indices.to_csv('df_X1X3_indices_k40.csv')
df_X1X3_loss.to_csv('df_X1X3_loss_k40.csv')
df_X2X3_indices.to_csv('df_X2X3_indices_k40.csv')
df_X2X3_loss.to_csv('df_X2X3_loss_k40.csv')



