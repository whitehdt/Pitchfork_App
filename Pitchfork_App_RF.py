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
from flask import Flask,render_template,request,redirect

os.chdir('/Users/drewwhitehead/Documents/The Data Incubator/Pitchfork Project/Flask App')
with open('X_train_mean_RF.pkl', 'rb') as input_file:
    X_predict = pickle.load(input_file)
with open('rf_gs.pkl', 'rb') as input_file:
    gs = pickle.load(input_file)

# create inputs lists and remove extranious items
inputList = list(X_predict)
removeItems = ['pub_year','word_count', 'char_count','avg_word','stopwords','sentiment','subjectivity','legacyReview','cluster']
for item in removeItems:
    inputList.remove(item)

# indicate cluster as "general"
X_predict['cluster'] = 1
X_predict['legacyReview'] = 0

# create flask obj
app_pitchfork = Flask(__name__)

# apply the class instance method (route) as the decorator
@app_pitchfork.route('/artist_info',methods=['GET','POST']) # 127.0.0.1:5000/artist_info
def artist_info():
    if request.method == 'GET':
        return render_template('artist_info_RF.html')
    else: # request was a post (meaning a user imput was provided)
        
        # zero out inputs
        for item in inputList:
            if item in X_predict:
                X_predict[item] = 0
        
        # add in new inputs
        for item in inputList:
            if item in request.form:
                X_predict[item] = request.form[item]

        prediction = gs.predict(X_predict)[0]
        prediction = round(prediction*100,2)
        
        return render_template('artist_pred_output_RF.html', prediction=prediction)

# triggers the debugger if the name is main
if __name__ == '__main__':
    app_pitchfork.run(debug=True)