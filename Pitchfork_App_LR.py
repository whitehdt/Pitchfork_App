#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:19:18 2018

@author: drewwhitehead
"""

# Drew Whitehead
# Pitchfork Project
# TDI Capstone Project - Flask App
# Python 3.6
import pickle
import os
from sklearn import linear_model
from flask import Flask,render_template,request,redirect

#os.chdir('/Users/drewwhitehead/Documents/The Data Incubator/Pitchfork Project/Flask App')
with open('/app/X_train_mean.pkl', 'rb') as input_file:
    X_predict = pickle.load(input_file)
with open('/app/lr_aba.pkl', 'rb') as input_file:
    lr_aba = pickle.load(input_file)

# create inputs lists and remove extranious items
inputList = list(X_predict)
removeItems = ['pub_year','word_count', 'char_count','sentiment','subjectivity','legacyReview']
for item in removeItems:
    inputList.remove(item)

# indicate cluster as "general"
X_predict['legacyReview'] = 0

# create flask obj
app_pitchfork = Flask(__name__)

# apply the class instance method (route) as the decorator
@app_pitchfork.route('/artist_info',methods=['GET','POST']) # 127.0.0.1:5000/artist_info
def artist_info():
    if request.method == 'GET':
        return render_template('artist_info_LR.html')
    else: # request was a post (meaning a user imput was provided)

        # zero out inputs
        for item in inputList:
            if item in X_predict:
                X_predict[item] = 0

        # add categorical inputs
        for item in inputList:
            if item in request.form:
                X_predict[item] = request.form[item]

        prediction = lr_aba.predict_proba(X_predict)[0][1]
        print(prediction)
        prediction = round(prediction*100,2)

        return render_template('artist_pred_output_LR.html', prediction=prediction)

# triggers the debugger if the name is main
if __name__ == '__main__':
    app.run(port=33507)
