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

#Import Imputer for missing value fill-in
from sklearn.preprocessing import Imputer
test_data_X = Imputer().fit_transform(test_data_X)
train_data_X = Imputer().fit_transform(train_data_X)
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


# Split into test/train test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data_X_pca, train_data_y, test_size = 0.2, random_state = 0)

'''
# Trial with Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, verbose = 10, n_jobs = -1)
'''

# Fitting XGBoost to the Training set
from xgboost import XGBRegressor
regressor = XGBRegressor()
'''
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

result = rmsle(y_pred, y_test)
'''


# Cross-validation with GridSearch
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

#Create the parameters list you wish to tune
parameters = {'learning_rate': [0.05, 0.1, 0.15],
              'min_child_weight': [1],
              'max_depth': [5, 7, 8],
              'subsample': [0.3, 0.5, 0.7],
              'objective': ["reg:linear"]
              }

scorer = make_scorer(rmsle, greater_is_better=False)

#Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = scorer,
                           cv = 10,
                           verbose=10,
                           n_jobs = -1)

#Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(train_data_X, train_data_y)


best_score = grid_fit.best_score_
best_parameters = grid_fit.best_params_

# Get the estimator
best_clf = grid_fit.best_estimator_

'''
regressor.fit(train_data_X, train_data_y)

'''

y_pred_test = best_clf.predict(test_data_X).astype(np.int32)


'''
y_pred_test = regressor.predict(test_data_X_pca).astype(np.int32)
'''

results = pd.DataFrame({
    'id' : test_data['id'].astype(np.int32),
    'price_doc' : y_pred_test
})

results.to_csv("./submission_with_XG_optimized_2.csv", index=False)

#######################################-----Day 2 Finsihed-----#########################################








