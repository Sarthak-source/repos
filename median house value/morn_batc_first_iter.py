import pandas as pd
#import datetime
#from datetime import datetime
# current date and time
#now = datetime.now()

df=pd.read_csv("case_study_preprocessing_L1.csv")
df.info()
df1=df.describe()

df.isnull().sum()
##################Removing entire row in df in which median_income is nan###########################
df.dropna(subset=['median_income'],inplace=True)
####################################################################################################

################## Filling nan values in total_bedrooms with median value ##########################
tb_med=df['total_bedrooms'].median()
df.fillna(tb_med,inplace=True)
###################################################################################################

df.isnull().sum()
df=df.reset_index()             #Resetting the index
df1=df.drop('index',axis=1)     #Removing the 'index' column

ax1 = df.plot.scatter(x='total_bedrooms',
                      y='median_income',
                      c='DarkRed')

boxplot = df.boxplot('latitude')
df['total_bedrooms'].quantile(0.95)

################### Removing outiers using IQR ####################################################
Q1 = df1.quantile(0.05)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1

index_outliers= df1.loc[((df1< (Q1 - 1.5 * IQR)) |(df1 > (Q3 + 1.5 * IQR))).any(axis=1)].index.tolist() #Index of the outliers 
outlier_values=df1.loc[index_outliers]    #List of values of outliers
remove_outliers=df1[~((df1< (Q1 - 1.5 * IQR)) |(df1 > (Q3 + 1.5 * IQR))).any(axis=1)]   # Removing outliers from dataframe and get the dataframe without outliers

df2=remove_outliers.reset_index()   # Resetting the index 
df_final=df2.drop('index',axis=1)
#y = datetime.datetime.now()
#timestamp = datetime.now()

################# encoding the categorical variables############

df1 = pd.get_dummies(df_final)
#now = time.strftime('%d%m%Y%H%M')
df1.to_csv('8_Nov_case1_clean_data_' +'.csv')

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


#linear regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


from sklearn.metrics import mean_squared_error
import numpy as np
y_pred = lin_reg.predict(X)
lin_mse = mean_squared_error(y, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

#lin r2 score

from sklearn.metrics import r2_score
r2_lin=r2_score(y,y_pred)

#rigde curve fit

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)

y_ridge = ridge_reg.predict(X)
lin_mse_ridge = mean_squared_error(y, y_ridge)
lin_rmse_ridge = np.sqrt(lin_mse_ridge)
lin_rmse_ridge


#rigde r2 score

from sklearn.metrics import r2_score
r2_ridge=r2_score(y,y_ridge)



#lasso curve fit

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)


y_lasso = lasso_reg.predict(X)
lin_mse_lasso = mean_squared_error(y, y_lasso)
lin_rmse_lasso = np.sqrt(lin_mse_lasso)
#lasso r2 score


r2_lasso=r2_score(y,y_lasso)



#ElasticNet curve fit

from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)

y_elastic_net = elastic_net.predict(X)
lin_mse_elastic_net = mean_squared_error(y, y_elastic_net)
lin_rmse_elastic_net = np.sqrt(lin_mse_elastic_net)
lin_rmse_elastic_net

#ElasticNet r2 score

r2_elastic=r2_score(y,y_elastic_net)

############## test data set ###############

y_ridge_test = ridge_reg.predict(X_test)
lin_mse_ridge_test = mean_squared_error(y_test, y_ridge_test)
lin_rmse_ridge_test = np.sqrt(lin_mse_ridge_test)
lin_rmse_ridge_test
r2_ridge_test=r2_score(y_test, y_ridge_test)

y_lin_test = lin_reg.predict(X_test)
lin_mse_lin_test = mean_squared_error(y_test, y_lin_test)
lin_rmse_lin_test = np.sqrt(lin_mse_lin_test)
lin_rmse_lin_test
r2_lin_test=r2_score(y_test, y_lin_test)



