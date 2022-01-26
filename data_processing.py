
# Importing Libraries

import numpy as np # for Arrays

import pandas as pd # for Data frame

import matplotlib.pyplot as plt # vistializatiom

# Importing data of KN house data for regrssion analysis

dataset = pd.read_csv(r"C:\Users\HP\Desktop\Data and AI\code practice\regression\kc_house_data.csv")

# Finding the Dependent variable and independent variable 

# Note for SImple linear regression one dependent variable and one independent variable we need to consider
# As per the data Price is derpendent variable and contineous nature so we need use regression model to predict best fit line

# Independent variable in data set 
area= dataset['sqft_living']

# Independent variable in data set
price = dataset['price']

# Assigning value for X and y 

X = np.array(area).reshape(-1,1)

y = np.array(price)

# Splitting the data into Train and Test

# Importing Sklearn model for anlysis 

from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 

regressor = LinearRegression()

regressor.fit(X_train, y_train)


#Predicting the prices
pred = regressor.predict(X_test)


#Visualizing the training Test Results 
plt.scatter(X_train, y_train, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

#Visualizing the Test Results 
plt.scatter(X_test, y_test, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()














