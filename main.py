#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:57:30 2017

@author: khchanaq
"""

import numpy as np
import pandas as pd
import visuals as vs
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor as rfr

#Import Imputer for missing value fill-in
from sklearn.preprocessing import Imputer

from sklearn.cross_validation import KFold

class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    
    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        
        folds = list(KFold(len(y), n_folds=self.n_folds, random_state=2017))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))
            print ("we are now in iteration : " + str(i))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred
        

#Convert all the string into float via hacky ways
def convert_to_float(dataset):

    for i in range(0,len(dataset[1])):
        for j in range(0, len(dataset)):
            if type(dataset[j,i]) is str:
                if dataset[j,i] == '#!':
                    dataset[j,i] = np.nan
                else:
                    dataset[j,i] = float(dataset[j,i].replace(',','').replace("-", ""))

    return dataset

def rmsle(predicted, actual):
    return np.sqrt(np.nansum(np.square(np.log(predicted + 1) - np.log(actual + 1)))/float(len(predicted)))

def FillMissingValue(df, fy):
    train_data = df[df.iloc[fy].notnull()]  
    test_data = df[df.iloc[fy].isnull()]  
    train_y=train_data.iloc[:,fy]
    train_X=train_data.copy()
    train_X.drop(train_X.columns[fy], axis = 1)
    test_X = test_data.copy()
    test_X.drop(test_X.columns[fy], axis =1)
    train_X = Imputer().fit_transform(train_X)
    test_X = Imputer().fit_transform(test_X)
    
    rfr_regressor=rfr(n_estimators=100,n_jobs=-1)
    #train the regressor
    rfr_regressor.fit(train_X,train_y)
    y_pred = rfr_regressor.predict(test_X)
    df[fy][df.iloc[fy].isnull()]= y_pred
    
    return df

# Cross-validation with GridSearch
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def AutoGridSearch(parameters, regressor):

    scorer = make_scorer(rmsle, greater_is_better=False)
#    while True:
    
    #Perform grid search on the classifier using 'scorer' as the scoring method
    grid_obj = GridSearchCV(estimator = regressor,
                               param_grid = parameters,
                               scoring = scorer,
                               cv = 10,
                               verbose=10,
                               n_jobs = -1)
    
    #Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_obj.fit(train_data_X, train_data_y)
    
    #if(best_score < grid_fit.best_score_):
    best_score = grid_fit.best_score_
        
        
    best_parameters = grid_fit.best_params_
    
    return best_score, best_parameters

def findMissingValue(X):
    #Check out Empty Data
    EmptyDataList = []
    for i in range(0, len(X[0])):
        if((np.isnan(np.min(X[:,i])))):
            EmptyDataList.append(i)

    return EmptyDataList



#Read dataset with pandas
train_data = pd.read_csv("train.csv", quoting = 2)
test_data = pd.read_csv("test.csv", quoting = 2)
macro_data = pd.read_csv("macro.csv", quoting = 2)

#Separate data into X,y
train_data_y = train_data.iloc[:,-1]
train_data_X = train_data.iloc[:,1:-1]
test_data_X = test_data.iloc[:,1:]
#join macro environment
train_data_X = pd.merge(train_data_X, macro_data, left_on='timestamp', right_index=True,
                  how='left', sort=False);
train_data_X = train_data_X.iloc[:,1:]
#join macro environment
test_data_X = pd.merge(test_data_X, macro_data, left_on='timestamp', right_index=True,
                  how='left', sort=False);
test_data_X = test_data_X.iloc[:,1:]

#One hot encoder for categorial variable
mixed_data_X = train_data_X.append(test_data_X, ignore_index=True)
length_train = len(train_data_X)
mixed_data_X = pd.get_dummies(mixed_data_X)

train_data_X = mixed_data_X.iloc[:length_train,:]
test_data_X = mixed_data_X.iloc[length_train:,:]

#Convert into Float for further operation
train_data_X = convert_to_float(train_data_X.values)
test_data_X = convert_to_float(test_data_X.values)

'''
train_data_X = pd.DataFrame(train_data_X)
test_data_X = pd.DataFrame(test_data_X)

NonNan_train = train_data_X.count()
NonNan_train = NonNan_train / len(train_data_X)
NonNan_test = test_data_X.count()
train_data_X = train_data_X.drop(train_data_X.columns[[3,4,5,6,7,8,20,143,144,145]], axis = 1)
test_data_X = test_data_X.drop(test_data_X.columns[[3,4,5,6,7,8,20,143,144,145]], axis = 1)
'''

#find MissingValue with return list of index of missing value
emptyList_train = findMissingValue(train_data_X)
emptyList_test = findMissingValue(test_data_X)

#set MissingValue with Random Forest
train_data_X = FillMissingValue(train_data_X)
test_data_X = FillMissingValue(test_data_X)

'''
corr_matrix =  (pd.DataFrame(train_data_X).corr())

for i in range (0, len(corr_matrix)):
    for j in range (0, len(corr_matrix)):
        if corr_matrix.iloc[i,j] == 1:
            corr_matrix.iloc[i,j] = 0

BestCorIndex = []
for i in EmptyDataList:
    print (i)
    print (np.max(corr_matrix.iloc[:,i-1]))
    index = np.argmax(corr_matrix.iloc[:,i-1])
    BestCorIndex.append([i,index])

BestCorIndex = pd.DataFrame(BestCorIndex).dropna().values

for i in range(0, 1):
    #Full Data Entry = Training Set
#    for j in range(0, len(train_data_X)):
    train_data_fillempty = train_data_X[:, [BestCorIndex[i][1].astype(np.int32), BestCorIndex[i][0].astype(np.int32)]]

        
    #Empty Data Entry = Testing Set
    test_data_X_fillempty = train_data_X[:, BestCorIndex[i] == np.nan]
    
'''

#test_data_X = Imputer().fit_transform(test_data_X)
#train_data_X = Imputer().fit_transform(train_data_X)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_data_X = sc_X.fit_transform(train_data_X)
test_data_X = sc_X.transform(test_data_X)
sc_y = StandardScaler()
train_data_y = sc_y.fit_transform(train_data_y)
'''
#######################################-----Day 1 Finsihed-----#########################################
'''
#feature reduction withPCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 6).fit(train_data_X)

train_data_X = pd.DataFrame(train_data_X)
test_data_X = pd.DataFrame(test_data_X)

# Generate PCA results plot
#pca_results = vs.pca_results(train_data_X, pca)

# Tranform into reduced data
train_data_X_pca = pca.transform(train_data_X)
test_data_X_pca = pca.transform(test_data_X)
'''
'''
# Split into test/train test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data_X_pca, train_data_y, test_size = 0.2, random_state = 0)
'''
'''
# Trial with Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, verbose = 10, n_jobs = -1)
'''

'''
# Setup BaseModel - XGboost
from xgboost import XGBRegressor as xgb
xgb_regressor = xgb(learning_rate = 0.0825, min_child_weight = 1, max_depth = 7, subsamples = 0.8, verbose = 10, n_jobs = -1)
Stacker_xgb_regressor = xgb(learning_rate = 0.0825, min_child_weight = 1, max_depth = 7, subsamples = 0.8, verbose = 10, n_jobs = -1)
'''

# Setup BaseModel - RandomForestRegressor
#from sklearn.ensemble import RandomForestRegressor as rfr
rfr_regressor = rfr(n_estimators = 200, verbose = 10, n_jobs = -1)

#Create the parameters list you wish to tune
rfr_parameters = {'max_features': [0.1, 0.5, 0.9],
              'min_samples_leaf': [50, 100, 150]}

rfr_best_score, rfr_best_parameters = AutoGridSearch(rfr_parameters,rfr_regressor)

from sklearn.ensemble import GradientBoostingRegressor as gbr
gbr_regressor = gbr(n_estimators = 200, verbose = 10)

#Create the parameters list you wish to tune
gbr_parameters = {'learning_rate': [0.1, 0.5, 1.0],
                  'max_depth': [3, 5, 7],
              'min_samples_leaf': [50, 100, 150],
              'subsample': [0.8],
              'max_features': [0.1, 0.5, 0.9]}

gbr_best_score, gbr_best_parameters = AutoGridSearch(gbr_parameters,gbr_regressor)

from sklearn.ensemble import ExtraTreesRegressor as etr
etr_regressor = etr(n_estimators = 200, verbose = 10)

#Create the parameters list you wish to tune
etr_parameters = {'learning_rate': [0.1, 0.5, 1.0],
                  'max_depth': [3, 5, 7],
              'min_samples_leaf': [50, 100, 150],
              'subsample': [0.8],
              'max_features': [0.1, 0.5, 0.9]}

etr_best_score, etr_best_parameters = AutoGridSearch(etr_parameters,etr_regressor)


# Trial with Random Forest
#ensemble = Ensemble(n_folds = 5,stacker =  Stacker_xgb_regressor,base_models = [xgb_regressor, rfr_regressor, gbr_regressor, etr_regressor])

#y_pred_test = ensemble.fit_predict(train_data_X, train_data_y, test_data_X)

#from sklearn.cross_validation import cross_val_score

#score = cross_val_score(ensemble, train_data_X, train_data_y, cv=2,scoring=rmsle)



'''
y_pred_test = regressor.predict(test_data_X_pca).astype(np.int32)
'''


'''
results = pd.DataFrame({
    'id' : test_data['id'].astype(np.int32),
    'price_doc' : y_pred_test
})

results.to_csv("./submission/submission_with_ensemble_1st_try_20170528_1.csv", index=False)
'''
#######################################-----Day 2 Finsihed-----#########################################


# Produce a scatter matrix for each pair of features in the data
#pd.scatter_matrix(macro_data.iloc[:, :], alpha = 0.3, figsize = (14,8), diagonal = 'kde');





