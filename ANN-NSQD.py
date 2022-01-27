# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 19:36:35 2022

@author: mmoein2
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
file_name = 'TSS'
data = pd.read_csv(file_name + '.csv', header=0, index_col=0)
analysis_type = input("Analysis Type 'R' or 'C': ")
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")
data_train, data_test = train_test_split(data, train_size = 0.8) #Only use stratify for classification
if analysis_type == 'R' or analysis_type == 'r':
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train_scaled = scaler.transform(data_train)
    X_train = data_train_scaled[:,0:-1] #Hacking a way to remove Grade since data_train_scaled is an array instead of a dataframe
    Y_train = data_train_scaled[:,-1]
    data_test_scaled = scaler.transform(data_test)
    X_test = data_test_scaled[:,0:-1]
    Y_test = data_test_scaled[:,-1]
if analysis_type == 'R' or analysis_type == 'r':
    acti = ['logistic', 'tanh', 'relu', 'identity']
    neural = MLPRegressor(activation=acti[3], solver='lbfgs', max_iter=200, hidden_layer_sizes=(10,10))
    neural_scores = cross_val_score(neural, X_train, Y_train, cv=5)
    print("Cross Validation Accuracy: {0} (+/- {1})".format(neural_scores.mean().round(2), (neural_scores.std() * 2).round(2)))
    print("")
    neural.fit(X_train, Y_train)
    neural_score = neural.score(X_test, Y_test)
    print("Shape of neural network: {0}".format([coef.shape for coef in neural.coefs_]))
    print("Coefs: ")
    print("")
    print(neural.coefs_[0].round(2))
    print("")
    print(neural.coefs_[1].round(2))
    print("")
    print("Intercepts: {0}".format(neural.intercepts_))
    print("")
    print("Loss: {0}".format(neural.loss_))
    print("")
    print("Iteration: {0}".format(neural.n_iter_))
    print("")
    print("Layers: {0}".format(neural.n_layers_))
    print("")
    print("Outputs: {0}".format(neural.n_outputs_))
    print("")
    print("Y test and predicted")
    print(Y_test.round(3))
    print(neural.predict(X_test).round(3))
    print("")
    print("Accuracy as Pearson's R2: {0}".format(neural_score.round(4)))
    print("")
    