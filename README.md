# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries and load the dataset (hours studied vs marks scored).
2. Split dataset into training and testing sets.
3. Train the model using Simple Linear Regression and predict results.
4. Evaluate performance and predict marks for new input.
5. 
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ANUBHARATHI SS
RegisterNumber:  212223040017

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head()

df.tail()

X = df.iloc[:, :-1].values
X

Y = df.iloc[:, 1].values
Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
Y_pred

Y_test

plt.scatter(X_train, Y_train, color="blue")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, regressor.predict(X_test), color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(Y_test, Y_pred)
print('MSE = ', mse)

mae = mean_absolute_error(Y_test, Y_pred)
print('MAE = ', mae)

rmse = np.sqrt(mse)
print("RMSE = ", rmse)
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
## HEAD:
<img width="291" height="245" alt="image" src="https://github.com/user-attachments/assets/a276e530-eda8-4001-adfe-1d99fae1c0bd" /> 

## TAIL:
<img width="264" height="246" alt="Screenshot 2025-08-24 190948" src="https://github.com/user-attachments/assets/3e0b311a-a72c-4d16-a614-dc8ede511b4f" />

## X:
<img width="309" height="561" alt="Screenshot 2025-08-24 191001" src="https://github.com/user-attachments/assets/00a78cbc-618f-4161-84ee-6cd04c8d52f4" />

## Y:
<img width="754" height="92" alt="Screenshot 2025-08-24 191016" src="https://github.com/user-attachments/assets/706f0ba8-191d-4f8f-927a-7e49ae0f225d" />

## Y_PRED:
<img width="714" height="69" alt="Screenshot 2025-08-24 191026" src="https://github.com/user-attachments/assets/54ae55b9-6714-4517-9499-44c4390c240f" />

## Y_TEST:
<img width="557" height="39" alt="Screenshot 2025-08-24 191035" src="https://github.com/user-attachments/assets/3d97a88a-ba3d-4e27-aa37-4abddb974fa1" />

## TRAINING SET:
<img width="786" height="583" alt="Screenshot 2025-08-24 191043" src="https://github.com/user-attachments/assets/b87a474c-8fbb-44cc-aa9b-03f9c8a3a761" />

## TEST SET:
<img width="730" height="582" alt="Screenshot 2025-08-24 191050" src="https://github.com/user-attachments/assets/6a2d54ff-9d73-44af-817b-98104bb937b2" />

## VALUES:
<img width="308" height="87" alt="Screenshot 2025-08-24 191055" src="https://github.com/user-attachments/assets/971c038b-952b-4ca8-89cf-989a2948dd17" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
