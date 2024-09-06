import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LassoCV


#----------------------------------------------------#
# Data Preparation 
#----------------------------------------------------#

df = pd.read_pickle('../../data/interim/03_Ames_Housing_Data_Encoded.pkl')

# Separating labels from the data

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Create train test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=101)

# Scale the data 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#----------------------------------------------------#
# Creating the model - Elastic Net
# ---------------------------------------------------#

# Create an elastic net model 

base_elastic_net = ElasticNet(max_iter=1000000)

# Set up a parameter grid

param_grid = {'alpha': [0.1, 1, 5, 10, 100],
              'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]}

# Create a grid search object

grid = GridSearchCV(base_elastic_net, param_grid, 
                    cv=5, scoring='neg_mean_squared_error',
                    verbose=1)

grid.fit(X_train_scaled, y_train)

grid.best_params_ # {'alpha': 100, 'l1_ratio': 1}

#----------------------------------------------------#
# Evaluating the model 
#----------------------------------------------------#

y_pred = grid.predict(X_test_scaled)

MAE =  mean_absolute_error(y_test, y_pred) # 14,195.32 
RMSE = np.sqrt(mean_squared_error(y_test, y_pred)) # 20,525.45 

#----------------------------------------------------#
# LassoCV Regression to attempt to simplify the model 
# ---------------------------------------------------#

lasso_cv_model = LassoCV(cv=5, max_iter=1000000)
lasso_cv_model.fit(X_train_scaled, y_train)

lasso_y_pred = lasso_cv_model.predict(X_test_scaled)

lasso_MAE =  mean_absolute_error(y_test, lasso_y_pred) # 14,184.51
lasso_RMSE = np.sqrt(mean_squared_error(y_test, lasso_y_pred)) # 20,524.42

lasso_cv_model.coef_

df_coef = pd.DataFrame(lasso_cv_model.coef_, index=X.columns, columns=['Coefficient'])

df_coef.sort_values(by='Coefficient', ascending=False).head(100)
df_coef[df_coef['Coefficient']==0].index

