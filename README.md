# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

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

Developed by: Leena shree M
RegisterNumber:  25018414

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Load dataset
data = pd.read_csv("/Salary.csv")

# Display basic info
data.head()
data.info()
data.isnull().sum()

# Encode categorical feature
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

# Define features and target
x = data[["Position", "Level"]]
x.head()
y = data["Salary"]
y.head()

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)

# Create and train model
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Predictions
y_pred = dt.predict(x_test)
y_pred

# Evaluate model
r2 = metrics.r2_score(y_test, y_pred)
print("R² Score:", r2)


```

## Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   Position  10 non-null     object
 1   Level     10 non-null     int64
 2   Salary    10 non-null     int64
dtypes: int64(2), object(1)
memory usage: 372.0+ bytes
R² Score: 0.48611111111111116


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
