import pandas as pd


df = pd.read_csv('8_Nov_case1_clean_data_.csv')
#df =df.reset_index(drop = True)

df =df.drop(df.columns[0],axis=1)


################# ML part #################################
#### split_train_test#################

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

X=train_set.drop('median_house_value',axis=1)
y=train_set['median_house_value']

X_test=test_set.drop('median_house_value',axis=1)
y_test=test_set['median_house_value']
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
import numpy as np


#polynomial degree=3 regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg_d_3=PolynomialFeatures(degree=3)
x_poly_3=poly_reg_d_3.fit_transform(X)
poly_reg_d_3.fit(x_poly_3,y)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly_3,y)

y_poly_perd=lin_reg2.predict(x_poly_3)
poly_3_mse = mean_squared_error(y, y_poly_perd)
poly_3_rmse= np.sqrt(poly_3_mse)
poly_3_rmse

#poly r2 score
r2_poly_3=r2_score(y, y_poly_perd)

################## test ######################

x_poly_3_t=poly_reg_d_3.fit_transform(X_test)
y_poly_perd_t=lin_reg2.predict(x_poly_3_t)
poly_3_mse_t = mean_squared_error(y_test, y_poly_perd_t)
poly_3_rmse_t= np.sqrt(poly_3_mse_t)
poly_3_rmse_t

#r2 test score 

r2_poly_3_t=r2_score(y_test, y_poly_perd_t)



