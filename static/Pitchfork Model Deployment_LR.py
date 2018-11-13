#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:38:10 2018

@author: drewwhitehead
"""
# *** refit model using logistic backwards elim ***
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

# Drew Whitehead
# Pitchfork Project
# TDI Capstone Project - Model Deployment
# Python 3.6

# load packages
import pandas as pd
import numpy as np
import os
from sklearn import model_selection, metrics, tree, linear_model
import pickle
import dill

# load modeling file
os.chdir('/Users/drewwhitehead/Documents/The Data Incubator/Pitchfork Project/Excel Files')
final_model_df = pd.read_csv('final_model_df.csv')
fitting_set = pd.read_csv('fitting_set.csv')
testing_set = pd.read_csv('testing_set.csv')
final_model_df = final_model_df.drop('Unnamed: 0', 1)
fitting_set = fitting_set.drop('Unnamed: 0', 1)
testing_set = testing_set.drop('Unnamed: 0', 1)


# convert categotical variables to dummy for FITTING
# er
fitting_set["established.rating"] = fitting_set["established.rating"].astype('category')
dict(enumerate(fitting_set["established.rating"].cat.categories)) # {0: 'Established', 1: 'New', 2: 'Semi-Established'}
fitting_set["established.rating"] = fitting_set["established.rating"].cat.codes
# genre
fitting_set["genre"] = fitting_set["genre"].astype('category')
dict(enumerate(fitting_set["genre"].cat.categories)) # {0: 'electronic', 1: 'experimental', 2: 'folk/country', 3: 'global', 4: 'jazz', 5: 'metal', 6: 'pop/r&b', 7: 'rap', 8: 'rock', 9: 'unlisted'}
fitting_set["genre"] = fitting_set["genre"].cat.codes
fitting_set["established.rating"].head(20)
fitting_set.dtypes

# convert categotical variables to dummy for TESTING
# er
testing_set["established.rating"] = testing_set["established.rating"].astype('category')
dict(enumerate(testing_set["established.rating"].cat.categories)) # {0: 'Established', 1: 'New', 2: 'Semi-Established'}
testing_set["established.rating"] = testing_set["established.rating"].cat.codes
# genre
testing_set["genre"] = testing_set["genre"].astype('category')
dict(enumerate(testing_set["genre"].cat.categories)) # {0: 'electronic', 1: 'experimental', 2: 'folk/country', 3: 'global', 4: 'jazz', 5: 'metal', 6: 'pop/r&b', 7: 'rap', 8: 'rock', 9: 'unlisted'}
testing_set["genre"] = testing_set["genre"].cat.codes
testing_set["established.rating"].head(20)
testing_set.dtypes



# create X & Y matrix
X_train = fitting_set[['char_count', 'word_count', 'genre', 'legacyReview', 
                 'spotifyPopularity', 'sentiment', 'pub_year', 'freak_folk', 
                 'art_pop', 'established.rating', 'emi_Label', 'warner_Label', 
                 'universal_Label', 'noise_rock', 'noise_pop', 'microhouse', 
                 'indie_rock', 'chillwave', 'drone', 'indietronica', 'chamber_psych', 
                 'indie_r_b', 'rock', 'modern_rock', 'electronic', 'indie_Label', 
                 'fourth_world', 'subjectivity']]
y_train = fitting_set['abv_blw_yr_avg_rat']

X_test = testing_set[['char_count', 'word_count', 'genre', 'legacyReview', 
                 'spotifyPopularity', 'sentiment', 'pub_year', 'freak_folk', 
                 'art_pop', 'established.rating', 'emi_Label', 'warner_Label', 
                 'universal_Label', 'noise_rock', 'noise_pop', 'microhouse', 
                 'indie_rock', 'chillwave', 'drone', 'indietronica', 'chamber_psych', 
                 'indie_r_b', 'rock', 'modern_rock', 'electronic', 'indie_Label', 
                 'fourth_world', 'subjectivity']]
y_test = testing_set['abv_blw_yr_avg_rat']

# fit linear regression model
lr_aba = linear_model.LogisticRegression()
lr_aba.fit(X_train, y_train)


# predict probability: train
y_probs = lr_aba.predict_proba(X_train)
y_pred = [1 if y[1] > 0.5 else 0 for y in y_probs]
print("Accuracy:", round(metrics.accuracy_score(y_train, y_pred),4)) # 0.7765
print("Precision:", round(metrics.precision_score(y_train, y_pred),4)) # 0.7637
print("Recall:", round(metrics.recall_score(y_train, y_pred),4)) # 0.8903


# predict probability: test
y_probs = lr_aba.predict_proba(X_test)
y_pred = [1 if y[1] > 0.5 else 0 for y in y_probs]
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred),4)) # 0.6338
print("Precision:", round(metrics.precision_score(y_test, y_pred),4)) # 0.653
print("Recall:", round(metrics.recall_score(y_test, y_pred),4)) # 0.7876

# create mean df for deployment
X_train_mean = pd.DataFrame(X_train.mean())
X_train_mean = X_train_mean.transpose()
list(X_train)

# pickle mean df and model
os.chdir('/Users/drewwhitehead/Documents/The Data Incubator/Pitchfork Project/Flask App/static')

# set recurse as true to enable the model to be run standalone
dill.settings['recurse'] = True

with open('X_train_mean.pkl', 'wb') as output_file:
    dill.dump(X_train_mean, output_file)
with open('lr_aba.pkl', 'wb') as output_file:
    dill.dump(lr_aba, output_file)

# ensure data can be extracted
with open('X_train_mean.pkl', 'rb') as input_file:
    X_train_mean = dill.load(input_file)
with open('lr_aba.pkl', 'rb') as input_file:
    lr_aba = dill.load(input_file)

import matplotlib.pyplot as plt
plt.hist(y_probs)



