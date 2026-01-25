import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

df=pd.read_csv("Breast_Cancer.csv")
print(df.head())

print(df.info())

print(df.columns)


print(df['Race'].value_counts())
print(df['Marital Status'].value_counts())
print(df['T Stage '].value_counts())
print(df['N Stage'].value_counts())
print(df['6th Stage'].value_counts())
print(df['differentiate'].value_counts())
print(df[ 'Grade'].value_counts())
print(df['A Stage'].value_counts())
print(df['Tumor Size'].value_counts())
print(df['Estrogen Status'].value_counts())
print(df['Progesterone Status'].value_counts())
print(df['Status'].value_counts())

df['Race']=df['Race'].map({'White':1 , 'Black':0 , 'Other':2})

df['Marital Status']=df['Marital Status'].map({'Married': 2 , 'Single': 0 , 'Divorced': -1 , 'Widowed': 1 , 'Separated': 0.5})

df['T Stage ']=df['T Stage '].map({'T1':1,'T2':2,'T3':3,'T4':4})

df['N Stage']=df['N Stage'].map({'N1':1,'N2':2,'N3':3})

df['6th Stage']=df['6th Stage'].map({'IIA':1,'IIB':2,'IIIA':3,'IIIC':4,'IIIB':5})

df['differentiate']=df['differentiate'].map({'Moderately differentiated':1 , 'Poorly differentiated':2 , 'Well differentiated':3 , 'Undifferentiated':4})

df['Grade']=df['Grade'].map({ ' well differentiated; Grade I': 1,
    ' moderately differentiated; Grade II': 2,
    ' poorly differentiated; Grade III': 3, ' anaplastic; Grade IV':4})


df['A Stage']=df['A Stage'].map({'Regional':1 , 'Distant':0})

df['Estrogen Status']=df['Estrogen Status'].map({'Positive':1 ,'Negative':0})

df['Progesterone Status']=df['Progesterone Status'].map({'Positive':1 ,'Negative':0})

df['Status']=df['Status'].map({'Alive':1 , 'Dead': 0 })

print(df.isnull().sum())

df['Marital Status'].fillna(df['Marital Status'].mode()[0] , inplace=True)

df['Grade'].fillna(df['Grade'].mode()[0] , inplace=True)

print(df.isnull().sum())

x=df.drop('Status' , axis=1)
y=df['Status']

x_train , x_test , y_train , y_test=train_test_split(x,y,test_size=0.2 , random_state=42)

dt_model=DecisionTreeClassifier(criterion="gini" , max_depth=4 , min_samples_leaf=20 , random_state=42)
dt_model.fit(x_train , y_train)

y_pred=dt_model.predict(x_test)

print("Accuracy of model : " , accuracy_score(y_test , y_pred))
print("classification of data : " , classification_report(y_test , y_pred))
print("confusion matrix of model : " , confusion_matrix(y_test , y_pred))

plt.figure(figsize=(40,25) , dpi=80)

plot_tree(dt_model , feature_names=x.columns , class_names=["Dead", "Alive"] , filled=True , rounded=True , fontsize=7)
plt.show()