#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:57:30 2017

@author: khchanaq
"""

#Basic Packages
import numpy as np
import pandas as pd

#Visualization for Data
import visuals as vs

#Import ML Models that used
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.ensemble import ExtraTreesRegressor as etr


#Import Imputer for missing value fill-in
from sklearn.preprocessing import Imputer

#StandardScaler for Feature Scaling
from sklearn.preprocessing import StandardScaler


#Feature Reduction with PCA package
from sklearn.decomposition import PCA


# Cross-validation with GridSearch
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold


#Ensemble Learning Model for final prediction - Stacking Approach
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

def fillMissingValue(df, fy):

    df = pd.DataFrame(df)
    train_data_temp = df[df.iloc[:,fy].notnull()]  
    test_data_temp = df[df.iloc[:,fy].isnull()]  
    train_y=train_data_temp.iloc[:,fy]
    train_X=train_data_temp.copy()
    train_X = train_X.drop(train_X.columns[fy], axis = 1)
    test_X = test_data_temp.copy()
    test_X = test_X.drop(test_X.columns[fy], axis =1)
    mixed_X = Imputer().fit_transform(train_X.append(test_X, ignore_index=True))
    length_train = len(train_X)
    train_X = mixed_X[:length_train,:]
    test_X = mixed_X[length_train:,:]
    
    print ("Try to fill-up value with rfr")
    rfr_regressor=rfr(n_estimators=100, verbose = 5, n_jobs = -1)
    #train the regressor
    rfr_regressor.fit(train_X,train_y)
    y_pred = rfr_regressor.predict(test_X)
    
    df[fy][df.iloc[:,fy].isnull()] = y_pred
    
    
    return df.values


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

def submit(test_data, y_pred, filename):

    results = pd.DataFrame({
    'id' : test_data['id'].astype(np.int32),
    'price_doc' : y_pred
    })

    results.to_csv("./submission/submission" + filename + ".csv", index=False)

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
templist = []
for i in range(0, len(mixed_data_X.iloc[0])):
    if((np.isnan(np.max(mixed_data_X.iloc[:,i])))):
        templist.append(i)
        
mixed_data_X = pd.DataFrame(mixed_data_X).drop(mixed_data_X.columns[templist], axis = 1)


train_data_X = mixed_data_X.iloc[:length_train,:]
test_data_X = mixed_data_X.iloc[length_train:,:]

#Convert into Float for further operation
train_data_X = convert_to_float(train_data_X.values)
test_data_X = convert_to_float(test_data_X.values)

#find MissingValue with return list of index of missing value
emptyList_train = findMissingValue(train_data_X)
emptyList_test = findMissingValue(test_data_X)


#TODO: Modulaize it

#set MissingValue with Random Forest
for i in emptyList_train:
    print ("Filling-up Train column : " + str(i))
    train_data_X = fillMissingValue(train_data_X, i)
    print ("The result of fill-up value = " + str(np.isnan(np.min(train_data_X[:,i]))))


for i in emptyList_test:
    print ("Filling-up Test column : " + str(i))
    test_data_X = fillMissingValue(test_data_X, i)
    print ("The result of fill-up value = " + str(np.isnan(np.min(test_data_X[:,i]))))

pd.DataFrame(train_data_X).to_csv("./ExtractedFeature/train_data_X_rfr_100.csv", index=False)

pd.DataFrame(test_data_X).to_csv("./ExtractedFeature/test_data_X_rfr_100.csv", index=False)


#TODO: future dig deep in feature extraction
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


#TODO: determine to have feature scaling or not

'''
# Feature Scaling
sc_X = StandardScaler()
train_data_X = sc_X.fit_transform(train_data_X)
test_data_X = sc_X.transform(test_data_X)
sc_y = StandardScaler()
train_data_y = sc_y.fit_transform(train_data_y)
'''
#######################################-----Day 1 Finsihed-----#########################################

#TODO: Consider to have feature reduction or not
    
'''
#feature reduction withPCA
pca = PCA(n_components = 6).fit(train_data_X)

train_data_X = pd.DataFrame(train_data_X)
test_data_X = pd.DataFrame(test_data_X)

# Generate PCA results plot
#pca_results = vs.pca_results(train_data_X, pca)

# Tranform into reduced data
train_data_X_pca = pca.transform(train_data_X)
test_data_X_pca = pca.transform(test_data_X)
'''

#TODO: Modulize the parameter tuning

# Setup BaseModel - XGboost
from xgboost import XGBRegressor as xgb
xgb_regressor = xgb(learning_rate = 0.0825, min_child_weight = 1, max_depth = 7, subsamples = 0.8, verbose = 10, random_state = 2017, n_jobs = -1)
Stacker_xgb_regressor = xgb(learning_rate = 0.0825, min_child_weight = 1, max_depth = 7, subsamples = 0.8, verbose = 10, random_state = 2017, n_jobs = -1)



# Setup BaseModel - RandomForestRegressor
rfr_regressor = rfr(n_estimators = 200, verbose = 10, max_features = 0.9, min_samples_leaf = 50, random_state = 2017)
# Best Parameter: max_features - 0.9, min_samples_leaf - 50 ; score -0.47517


#Create the parameters list you wish to tune
rfr_parameters = {'max_features': [0.7, 0.8, 0.9],
              'min_samples_leaf': [25, 50, 75],
              'random_state': [2017]}

#rfr_best_score, rfr_best_parameters = AutoGridSearch(rfr_parameters,rfr_regressor)

gbr_regressor = gbr(n_estimators = 200, verbose = 5, learning_rate = 0.1, max_depth = 7, max_features = 0.5, min_samples_leaf = 50, subsample = 0.8, random_state = 2017)
# Best Parameter: learning_rate - 0.1, max_depth = 7, max_features - 0.5, min_samples_leaf - 50, sub_samples = 0.8 ; score -0.466787


#Create the parameters list you wish to tune
gbr_parameters = {'learning_rate': [0.01, 0.1, 0.5],
                  'max_depth': [6, 7, 8],
              'min_samples_leaf': [25, 50, 75],
              'subsample': [0.8],
              'max_features': [0.3, 0.5, 0.7],
              'random_state': [2017]}

#gbr_best_score, gbr_best_parameters = AutoGridSearch(gbr_parameters,gbr_regressor)

etr_regressor = etr(n_estimators = 200, verbose = 10, max_depth = 7, min_samples_leaf = 100, max_features = 0.9, min_impurity_split = 100, random_state = 2017)

#Create the parameters list you wish to tune
etr_parameters = {'max_depth': [3, 5, 7],
              'min_samples_leaf': [50, 100, 150],
              'max_features': [0.1, 0.5, 0.9],
              'min_impurity_split': [50, 100, 150],
              'random_state': [2017]}

#etr_best_score, etr_best_parameters = AutoGridSearch(etr_parameters,etr_regressor)


#TODO: Ensemble with CV score implementation

# Ensemble Run
ensemble = Ensemble(n_folds = 5,stacker =  Stacker_xgb_regressor,base_models = [xgb_regressor, rfr_regressor, gbr_regressor, etr_regressor])

y_pred = ensemble.fit_predict(train_data_X, train_data_y, test_data_X)

xgb_regressor.fit(train_data_X, train_data_y)

y_pred = xgb_regressor.predict(test_data_X)

submit(test_data, y_pred, "Ensemble-V1")




