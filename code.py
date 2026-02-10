#data analysis
import pandas as pd
import matplotlib.pyplot as plt
siri=pd.read_csv("Flipkart_mobile_brands_scraped_data.csv")
# df=siri["Color"].fillna("black" , inplace=True)
df=siri[siri.isnull().any(axis=1)]

siri.loc[20,"Original Price"]=123.00

print(siri.loc[20,"Original Price"])

df=siri[siri.isnull().any(axis=1)]

# siri.drop([0,1,2] , inplace=True)
# siri['London']=list(range(1,len(siri)+1))
siri.loc[len(siri)]=["apple","12","white",'4GB','64GB',4.5,119990,1400]
print(siri)
new_row = pd.DataFrame([{
    "Brand": "OnePlus",
    "Model": "Nord",
    "Color": "Blue",
    "Original Price": 30000,
    "Selling Price": 28000
}])

siri = pd.concat([siri, new_row], ignore_index=True)
print(siri)  # ignore index for to create a new continuous index

def sumofcol(x) :
    a=x.sum()
    return a

b=sumofcol(siri['Original Price'])
print(b)


df=siri.plot.bar(y="Original Price" , title="bar chat", xlabel="Brand" , ylabel="Model")
plt.show()
print(df)
# a=[]
# print(len(siri[siri["Brand"]=='Apple']))














import pandas as pd
import matplotlib . pyplot as plt
import numpy as np
import seaborn as sns

df=pd.read_csv("Flipkart_mobile_brands_scraped_data.csv")
def detect_outliers_iqr(df, column) :
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR=Q3-Q1

    lower_bound=Q1-1.5*IQR
    upper_bound=Q3 + 1.5*IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

outliers=detect_outliers_iqr(df,"Selling Price")
print("number of Outliers : " , len(outliers))
print(outliers)

# def clean_outliers_iqr(df, column) :
#     Q1=df[column].quantile(0.25)
#     Q3=df[column].quantile(0.75)
#     IQR=Q3-Q1
                                             #FOR REMOVE THE OUTLIERS
#     lower_bound=Q1-1.5*IQR
#     upper_bound=Q3 + 1.5*IQR

#     df_clean = df[(df[column] <= lower_bound) & (df[column] >= upper_bound)]
#     return df_clean

# df=clean_outliers_iqr(df,"Selling Price")
# print("Number of outliers : " , len(df)) 

plt.figure(figsize=(8,4))
sns.boxplot(x=df["Selling Price"])
plt.title("Outlier detection - Selling price")
plt.show()


#LINEARREGRESSION SESON 5

import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
from sklearn.linear_model import LinearRegression

#create dummy data
#year of experience
x=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)

#salary corresponding to experience
y=np.array([30000,35000,40000,45000,50000,55000,60000,65000,70000,75000])

#create and train the model
model=LinearRegression() #y=mx+c
model.fit(x,y) #training ( 80 percent),testing(20 percent) fit la mtlab hai sikhana apne model ko 

#make predictions
y_pred=model.predict(x) 
#print model parameters how your model is working 
print("slope(cofficient) : m" , model.coef_[0])
print("Intercept : c" , model.intercept_)

#plot the result
plt.scatter(x,y,label="Actual salary") #scatter means : graph with dot (ek line bnega usper dot me x and y ko show kerega)
plt.plot(x,y_pred,color="red", label="LinearRegression")
 # x → Years of experience (input values) , y_pred → Predicted salary (output from the model)
 #color : It makes the line red


plt.xlabel("Year of experience")
plt.ylabel("Salary")
plt.title("Linear Regression : Salary vs experience")
plt.legend() #it show my values which is at corner
plt.show()




#LOGISTIC REGRESSION SEASON 7

# ================================
# 1. Import required libraries
# ================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ================================
# 2. Load the dataset
# ================================
df = pd.read_csv("Heart_Disease_Prediction.csv")

print("First 5 rows of data:")
print(df.head())


# ================================
# 3. Check dataset information
# ================================
print("\nDataset Info:")
print(df.info()) #Check structure, data type, missing values

print("\nTarget column values:")
print(df["Heart Disease"].value_counts())
# "We use df.info() to understand dataset structure and value_counts() to check class balance before training the model."


# ================================
# 4. Convert target column to numeric
# Presence  -> 1
# Absence   -> 0
# ================================
df["Heart Disease"] = df["Heart Disease"].map({
    "Presence": 1,
    "Absence": 0
}) #Machine Learning models DO NOT understand text so we are converting presence and absence in 1 and 0
 

# ================================
# 5. Split features and target
# ================================
X = df.drop("Heart Disease", axis=1)  # Input features
y = df["Heart Disease"]               # Output (target)
# X contains independent variables (features), y contains dependent variable (target).

# ================================
# 6. Train-Test Split
# 80% training, 20% testing
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train_test_split	Split data for training & testing
# test_size=0.2	    20% testing data
# random_state=42	Same result every time

# ================================
# 7. Create Logistic Regression Model
# ================================
model = LogisticRegression(max_iter=1000)


# ================================
# 8. Train the model
# ================================
model.fit(X_train, y_train)
#X_train → Questions
#y_train → Answer sheet
#fit() → Teaching process

# ================================
# 9. Make predictions
# ================================
y_pred = model.predict(X_test)
# test data ke liye model ka output batata hai.

# ================================
# 10. Model Evaluation
# ================================
accuracy = accuracy_score(y_test, y_pred)
# accuracy_score()	Calculate correct prediction rate
# y_test	        Actual answers
# y_pred	        Predicted answers
print("\nAccuracy of the model:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ================================
# 11. Predict for a new patient
# ================================
sample_data = pd.DataFrame([[63, 1, 4, 145, 233, 1, 0, 150, 0, 2.3, 3, 0, 6]], columns=x.columns)
prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("\nHeart Disease: Presence")
else:
    print("\nHeart Disease: Absence")



#DRUG FILE  WRITE LOGISTICREGRESSION 


df=pd.read_csv("drug200.csv")
print(df)

df['Drug']=df['Drug'].replace('DrugY' , 'drugY')

print(df["Sex"].value_counts())
df["Sex"]=df["Sex"].replace('M','0')
df["Sex"]=df['Sex'].replace('F' , '1')

df['Sex']=df['Sex'].astype(int)

print(df['BP'].value_counts())

df['BP']=df['BP'].replace('HIGH' , '1')
df['BP']=df['BP'].replace('LOW' , '1')
df['BP']=df['BP'].replace('NORMAL' , '2')

df['BP']=df['BP'].astype(int)

print(df['Cholesterol'].value_counts())

df["Cholesterol"]=df["Cholesterol"].replace('HIGH' , '0')
df["Cholesterol"]=df["Cholesterol"].replace('NORMAL' , '0')

print(df['Drug'].value_counts())

df['Drug']=df['Drug'].replace('drugY' , '0')
df['Drug']=df['Drug'].replace('drugX' , '1')
df['Drug']=df['Drug'].replace('drugA' , '2')
df['Drug']=df['Drug'].replace('drugC' , '3')
df['Drug']=df['Drug'].replace('drugB' , '4')

print(df)

x=df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y=df['Drug']

x_train , x_test , y_train , y_test=train_test_split(x,y,test_size=0.4,random_state=42)

model=LogisticRegression(max_iter=2000)
model.fit(x_train , y_train)

y_pred=model.predict(x_test)

print("accuracy : " , accuracy_score(y_test , y_pred))
print("classification : " , classification_report(y_test , y_pred))
print("counfision matrix : " , confusion_matrix(y_test , y_pred))













#SEASONE 8
# first we add data then we check null values of not if yes then fill
  # them and in logistic rergression it works for ony numerical values so ve convert into numerical values
# then we check accuracy classification and confusion and then we plot the graph of my categorized features

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")
print(df)

df = df.drop("Loan_ID", axis=1)


print('My first 5 data set : ')
print(df.head())

print('infortmation of my data : ')
print(df.info())

print(df.isnull().sum())

# Fill numeric missing values
df["LoanAmount"].fillna(df["LoanAmount"].mean(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

# Fill categorical missing values
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)


print(df["Gender"].value_counts())
print(df["Education"].value_counts())
print(df["Property_Area"].value_counts())

df["Gender"]=df["Gender"].replace('Male' , 1)
df["Gender"]=df["Gender"].replace('Female' , 0)

df["Married"]=df["Married"].replace('No' , 0)
df["Married"]=df["Married"].replace('Yes' , 1)


df["Loan_Status"]=df["Loan_Status"].replace('No' , 0)
df["Loan_Status"]=df["Loan_Status"].replace('Yes' , 1)

df["Self_Employed"]=df["Self_Employed"].replace('No' , 0)
df["Self_Employed"]=df["Self_Employed"].replace('Yes' , 1)

df["Education"]=df["Education"].replace("Graduate" , 1)
df["Education"]=df["Education"].replace("Not Graduate" , 0)

df["Property_Area"]=df["Property_Area"].replace("Semiurban" , 1)
df["Property_Area"]=df["Property_Area"].replace("Urban" , 0)
df["Property_Area"]=df["Property_Area"].replace("Rural" , 2)


# for remove some warning given by pnadas we do istead of replace we will use map
df["Dependents"] = df["Dependents"].replace("3+", 3)
df["Dependents"] = df["Dependents"].astype(int)



x=df.drop("Loan_Status" , axis=1)
y=df["Loan_Status"] 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(x_train , y_train)

y_pred=model.predict(x_test)

print("Accuracy : " , accuracy_score(y_test,y_pred))
print("classification : " , classification_report(y_test , y_pred))
print("confusion matrix : " , confusion_matrix(y_test , y_pred))

categorized_features=['Gender' , 'Married' , 'Education' , 'Self_Employed' , 'Property_Area' , 'Credit_History' , 'Loan_Status']

plt.figure(figsize=(15 ,10))
for i,feature in enumerate(categorized_features , 1 ):
    
    plt.subplot( 3 ,3 ,i)
    sns.countplot(x=feature , data=df , palette="viridis")
    plt.title(f'distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

#NOW WE WILL MAKE THE GRAPH FOR MUMERICAL_FEATURES

numerical_feature=['ApplicantIncome' , 'CoapplicantIncome' , 'LoanAmount' , 'Loan_Amount_Term']
plt.figure(figsize=(15 ,12))
for i,feature in enumerate(numerical_feature , 1):
    plt.subplot(4,2,2*i-1) #Histogram
    sns.histplot(df[feature].dropna() , kde=True , palette="viridis")
    plt.title(f'distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.subplot(4,2,2*i)
    sns.boxplot(x=df[feature].dropna() , palette="viridis")
    plt.title(f'Box plot of {feature}')
    plt.xlabel(feature)
    plt.tight_layout()
    plt.show()

    #NEW QUESTION FOR LOGESTIC REGRESION

    import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

model=LogisticRegression(max_iter=1000)
model.fit(x_train , y_train)

y_pred=model.predict(x_test)

print("Accuracy of model : " , accuracy_score(y_test , y_pred))
print("classification of model : " , classification_report(y_test , y_pred))
print("Confusion matrix : " , confusion_matrix(y_test , y_pred))

print(df.dtypes)

categorized_feature=['Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage',
       'differentiate', 'Grade', 'A Stage', 'Tumor Size', 'Estrogen Status',
       'Progesterone Status']
plt.figure(figsize=(20,12))
for i,feature in enumerate(categorized_feature , 1):
     plt.subplot(4,4,i)
     sns.countplot(x=feature , data=df , palette="viridis")
     plt.title(f'Distribution of {feature}')
     plt.xlabel(feature)
     plt.ylabel('count')
     plt.tight_layout
     plt.show()

numerical_feature=['Age' ,'Regional Node Examined',
                   'Reginol Node Positive', 'Survival Months', 'Status' ]

for i,feature in enumerate(numerical_feature , 1):
     plt.subplot(5,2,2*i-1)
     sns.histplot(df[feature].dropna() , kde=True , palette="viridis")
     plt.title(f'Distribution of {feature}')
     plt.xlabel(feature)
     plt.ylabel('count')

     plt.subplot(5,2,2*i)
     sns.boxplot(x=df[feature].dropna() , palette="viridis")
     plt.title(f'Box plot of {feature}')
     plt.xlabel(feature)
     plt.tight_layout()
     plt.show()








#SEASON 9 LEARN ABOUT DECISION TREE < RANDOM FOREST < XGBOOST 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix

df=pd.read_csv("mobile price classification dataset.csv")
print(df.head())

print(df.isnull().sum())

for column in df.columns:
    unique_values_count=df[column].nunique()
    print(f"column '{column}' : {unique_values_count} unique values")
    if(unique_values_count <= 10):
        print("unique value :" , df[column].unique())
    else:
        print("\n")

print(df.dtypes)
print(df.columns)

y = df['four_g']      # predict 5G support
x = df.drop('four_g', axis=1)

x_train , x_test , y_train , y_test=train_test_split(x,y,test_size=0.2,random_state=42)
dt_model=DecisionTreeClassifier(criterion="gini" ,  class_weight="balanced",max_depth=3 ,  min_samples_leaf=20 , random_state=42)
dt_model.fit(x_train , y_train)

print("My dt_model is succesfully tarined")

y_pred=dt_model.predict(x_test)


print("Accuracy of model : " , accuracy_score(y_test , y_pred))
print("classification of data : " , classification_report(y_test , y_pred))
print("confusion matrix of model : " , confusion_matrix(y_test , y_pred))

plt.figure(figsize=(25,14) , dpi=200)

plot_tree(dt_model , feature_names=x.columns , rounded=True , filled=True , fontsize=6 )
plt.show()

rf_model=RandomForestClassifier( n_estimators=300, class_weight="balanced", max_features="sqrt" , min_samples_leaf=3 , max_depth=12 , n_jobs=-1
, random_state=42)
rf_model.fit(x_train , y_train)

y_pred=rf_model.predict(x_test)

print("Accuracy of model : " , accuracy_score(y_test , y_pred))
print("classification of data : " , classification_report(y_test , y_pred))
print("confusion matrix of model : " , confusion_matrix(y_test , y_pred))

xgb_model=XGBClassifier( random_state=42)
xgb_model.fit(x_train , y_train)

print("XGB model trained succesfully")
y_pred=xgb_model.predict(x_test)

print("Accuracy of model : " , accuracy_score(y_test , y_pred))
print("classification of data : " , classification_report(y_test , y_pred))
print("confusion matrix of model : " , confusion_matrix(y_test , y_pred))
