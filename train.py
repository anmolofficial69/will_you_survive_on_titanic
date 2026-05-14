import pandas as pd
import numpy as np
df=pd.read_csv('synthetic_titanic_dataset.csv')
df['Sex']=df['Sex'].map({'male':0,'female':1})
X=df[['Sex', 'Age', 'Pclass', 'Fare', 'SibSp', 'Parch']]
y=df['Survived']
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=XGBClassifier(n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    n_jobs=-1)

model.fit(X_train,y_train)

a=input("Enter your gender (male) or (female) : ")
if a=='male' or a=='Male' or a=='MALE':
  a=0
elif a=='female' or a=='Female'or a=='FEMALE':
  a==1

b=int(input("Enter your age : "))
c=int(input("Enter your class 1/2/3 : "))
d=float(input("Enter your fare : "))
e=int(input("Enter number of siblings : "))
f=int(input("Enter number parents : "))
y_pred=model.predict([[a,b,c,d,e,f]])
if(y_pred==1):
  print("Survived")
else:
  print("Not Survived")