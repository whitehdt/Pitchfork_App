#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:38:10 2018

@author: drewwhitehead
"""

# Drew Whitehead
# Pitchfork Project
# TDI Capstone Project - Model Deployment
# Python 3.6

# load packages
import pandas as pd
import numpy as np
import os
from sklearn import model_selection, ensemble, metrics, tree
import pickle

# load modeling file
os.chdir('/Users/drewwhitehead/Documents/The Data Incubator/Pitchfork Project/Excel Files')
final_model_df = pd.read_csv('final_model_df.csv')
final_model_df = final_model_df.drop('Unnamed: 0', 1)

# convert categotical variables to dummy
# er
final_model_df["established.rating"] = final_model_df["established.rating"].astype('category')
dict(enumerate(final_model_df["established.rating"].cat.categories)) # {0: 'Established', 1: 'New', 2: 'Semi-Established'}
final_model_df["established.rating"] = final_model_df["established.rating"].cat.codes
# genre
final_model_df["genre"] = final_model_df["genre"].astype('category')
dict(enumerate(final_model_df["genre"].cat.categories)) # {0: 'electronic', 1: 'experimental', 2: 'folk/country', 3: 'global', 4: 'jazz', 5: 'metal', 6: 'pop/r&b', 7: 'rap', 8: 'rock', 9: 'unlisted'}
final_model_df["genre"] = final_model_df["genre"].cat.codes
final_model_df["established.rating"].head(20)
final_model_df.dtypes

# create X & Y matrix
X = final_model_df.drop(['abv_blw_yr_avg_rat','reviewid','artist','best_new_music','score'], axis=1)
y = final_model_df['abv_blw_yr_avg_rat']

# divide into train & test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)
X.shape, X_train.shape, X_test.shape

# fit decision tree
est = ensemble.RandomForestRegressor()
gs = model_selection.GridSearchCV(
    est,
    {"n_estimators": [300, 600, 900],
     "max_depth": [2, 5, 10],
     "min_samples_split": [5]},
    cv=5,  # 5-fold cross validation
    n_jobs=2,  # run each hyperparameter in one of two parallel jobs
    scoring='neg_mean_squared_error' # opposite of MSE is a utility funtion which we are trying to maximize vs minimize
)
gs.fit(X_train, y_train)
print (gs.best_params_) # {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 900}


# predict probability: train
y_probs = gs.predict(X_train)
y_pred = [1 if y > 0.5 else 0 for y in y_probs]
print("Accuracy:", round(metrics.accuracy_score(y_train, y_pred),4)) # 0.7765
print("Precision:", round(metrics.precision_score(y_train, y_pred),4)) # 0.7637
print("Recall:", round(metrics.recall_score(y_train, y_pred),4)) # 0.8903


# predict probability: test
y_probs = gs.predict(X_test)
y_pred = [1 if y > 0.5 else 0 for y in y_probs]
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred),4)) # 0.6338
print("Precision:", round(metrics.precision_score(y_test, y_pred),4)) # 0.653
print("Recall:", round(metrics.recall_score(y_test, y_pred),4)) # 0.7876

# create mean df for deployment
X_train_mean_RF = pd.DataFrame(X_train.mean())
X_train_mean_RF = X_train_mean_RF.transpose()

# pickle mean df and model
os.chdir('/Users/drewwhitehead/Documents/The Data Incubator/Pitchfork Project/Flask App')
with open('X_train_mean_RF.pkl', 'wb') as output_file:
    pickle.dump(X_train_mean_RF, output_file)
with open('rf_gs.pkl', 'wb') as output_file:
    pickle.dump(gs, output_file)
# ensure data can be extracted
with open('X_train_mean_RF.pkl', 'rb') as input_file:
    X_train_mean_RF = pickle.load(input_file)
with open('rf_gs.pkl', 'rb') as input_file:
    gs = pickle.load(input_file)

import matplotlib.pyplot as plt
plt.hist(y_probs)

X_train.dtypes

