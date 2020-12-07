# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:55:01 2020

@author: Sai Ram. K
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
df = pd.read_csv(r'C:\Users\ram10\Desktop\Vth SEM\VAC -ML\diab.csv')
print(df.head(5),'\n\n')
X = df[[ 'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df[['Outcome']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
clf = KNeighborsClassifier(n_neighbors=25)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print('done')
pickle.dump(clf, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,135,75,29,0,33.6,0.700,45]]))