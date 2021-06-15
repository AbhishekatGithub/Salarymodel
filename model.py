
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import math

# load the dataset
salary = pd.read_csv('Salary_Data.csv')
print(pd.DataFrame(salary))
x= salary.iloc[:, :-1].astype('float').values  
y= salary.iloc[:, 1].astype('float').values  
print(x)
print("_________")
print(y)

# splitting X and y into training and testing sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
 
# create linear regression object
regressor =LinearRegression()
 
# train the model using the training sets
regressor.fit(x_train, y_train)

#print('Coefficients: ', regressor.coef_)
#val=3
#preds=np.array(val)
#preds=preds.reshape((1,-1))


#print("Prediction:", math.floor(regressor.predict(preds)))
#variance score: 1 means perfect prediction
#print('Variance score: {}'.format(regressor.score(x_test, y_test)))
pickle.dump(regressor, open('model.pkl', 'wb'))
#model = pickle.load(open('model.pkl', 'rb'))
#print("Predicted value is:",model.predict([[2]]))