# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## Aim:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values. 


## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SWATHI D
RegisterNumber: 212222230154

import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head() 
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_predict)
mse
r2=metrics.r2_score(y_test,y_predict)
r2
dt.predict([[6,7]])

```

## Output:

### Initial Dataset:
![ml exp7-1](https://github.com/Gopika-9266/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122762773/a1ead9c7-0191-4d1e-b063-e596edee8bf7)

### data.info():
![ml exp7-2](https://github.com/Gopika-9266/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122762773/3d6ebe83-e34e-4ce1-ae4d-d6db8d189cee)

### Optimisation of Null values:
![ml exp7-3](https://github.com/Gopika-9266/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122762773/12cf41be-09ee-46b4-8576-04bdbf8b149b)

### Converting string literals to numerical values using label encoder:
![ml exp7-4](https://github.com/Gopika-9266/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122762773/2f4509a8-7067-4748-8a3f-64880cf3da7e)

### Mean Squared Error:
![ml exp7-5](https://github.com/Gopika-9266/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122762773/41b8660a-3a2e-4753-916a-7cbb0e94b5c7)

### R2 (variance):
![ml exp7-6](https://github.com/Gopika-9266/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122762773/e85396b2-e9fb-4584-8457-4233d3ddd05e)

### Data prediction:
![ml exp7-7](https://github.com/Gopika-9266/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122762773/bc8bb3c3-abce-449f-85cd-343e08680631)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
