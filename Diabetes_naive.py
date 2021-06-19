# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 21:00:26 2021

@author: Soumya PC
"""

import pandas as pd
import numpy as np
Diabetes = pd.read_csv("Diabetes_RF.csv")
colnames = list(Diabetes.columns)
colnames
predictors = colnames[0:8]
target = colnames[8]
from sklearn.model_selection import train_test_split
#splitting the data into train and test
DXtrain,DXtest,Dytrain,Dytest = train_test_split(Diabetes[predictors],Diabetes[target],test_size=0.3, random_state=0)

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
building the 2 models
Dgnb = GaussianNB()
Dmnb = MultinomialNB()

building the model and predicting at the same time
Dpred_gnb = Dgnb.fit(DXtrain,Dytrain).predict(DXtest)
Dpred_mnb = Dmnb.fit(DXtrain,Dytrain).predict(DXtest)
pd.crosstab(Dpred_gnb,Dytest)
from sklearn.metrics import confusion_matrix
print('Accuracy',(138+38)/(138+38+19+36))

pd.crosstab(Dpred_mnb,Dytest)
print('Accuracy',(114+36)/(114+36+43+38))
