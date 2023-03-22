# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings('ignore')

# Importing the preprocessed Salary Dataset
data=pd.read_csv("C:\\Users\\adilv\\Desktop\\Internship project\\Preprocessed salarydata.csv")

# Splitting X and y into features and target...y is the dependent variable
X=data.drop('salary',axis=1)
y=data['salary']

# Splitting the data into train and test
import sklearn
from sklearn.model_selection import train_test_split

# Model
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=.2)

# Creating Gradient Boosting Classification Model
from sklearn.ensemble import GradientBoostingClassifier
grad_boost_model=GradientBoostingClassifier().fit(X_train,y_train)
# Predicting the Test set Result
y_pred_grad=grad_boost_model.predict(X_test)


# Fine Tuning using RandomizedSearchCV
model_random=GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.1)
model2=model_random.fit(X_train,y_train)
y_pred_random=model2.predict(X_test)

# Training the preprocessed data with the best Hyperparameters
model_random=GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.1)
model_random.fit(X,y)

# Saving model using pickle
pickle.dump(model_random,open('randomcv_grad_model.pkl','wb'))

# Loading model to compare the result
model=pickle.load(open('randomcv_grad_model.pkl','rb'))

print(model.predict([[39.0,6,0,1,0,1,4,1,40.0,38]]))
