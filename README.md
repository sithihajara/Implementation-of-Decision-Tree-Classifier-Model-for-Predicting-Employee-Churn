# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sithi hajara I
RegisterNumber:  212221230102
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![2006![200617442-2d8a2866-06b5-4717-952b-c75a55b73886](https://user-images.githubusercontent.com/94219582/200753424-879d60fe-8aab-42d9-a7a6-7a158577bb34.png)
17343-4b606eaf-6918-46b1-a7ef-91a49bf4e4fc](https://user-images.githubusercontent.com/94219582/200753399-43cf214d-aa99-4cbf-9c10-d74fabab5186.png)
![200617478-f81bd031-ccdd-4d22-8295-758384740aa5](https://user-images.githubusercontent.com/94219582/200753469-6c5408a7-2ac8-408d-860d-76877adabf34.png)
![200617600-5af99405-5076-49df-b9e7-e3ea8696d097](https://user-images.githubusercontent.com/94219582/200753493-276672ee-7fd1-43a0-8836-8b1033d14692.png)
![200617634-e069520e-33b3-4015-8a77-4cfe9c47b53c](https://user-images.githubusercontent.com/94219582/200753510-105cbde3-e162-4981-8fbc-59e610f3c979.png)
![200617666-daff2eab-0306-453c-94c6-4456beb9565d](https://user-images.githubusercontent.com/94219582/200753529-88a3290a-70ce-4bbc-b909-fee6b8a32f25.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
