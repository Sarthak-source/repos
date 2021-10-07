import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Importing the dataset
dataset = pd.read_csv('Placement_Data_Full_Class.csv')



dataset.shape
dataset.info()
dataset.describe()

dataset.workex.replace(('Yes','No'),(True,False),inplace=True)
dataset.status.replace(('Placed','Not Placed'),(True,False),inplace=True)
dataset.salary.replace(np.nan,0,inplace=True)

def Encode(dataset):
    for column in dataset.columns[~dataset.columns.isin([ 'ssc_p', 'hsc_p','degree_p', 'etest_p', 'mba_p', 'salary'])]:
        dataset[column] = dataset[column].factorize()[0]
    return dataset

dataset_en = Encode(dataset.copy())

Q1 = dataset_en.quantile(0.05)
Q3 = dataset_en.quantile(0.80)
IQR = Q3 - Q1


index_outliers= dataset_en.loc[((dataset_en< (Q1 - 1.5 * IQR)) |(dataset_en > (Q3 + 1.5 * IQR))).any(axis=1)].index.tolist() #Index of the outliers 
outlier_values=dataset_en.loc[index_outliers]    #List of values of outliers
remove_outliers=dataset_en[~((dataset_en< (Q1 - 1.5 * IQR)) |(dataset_en > (Q3 + 1.5 * IQR))).any(axis=1)]   # Removing outliers from dataframe and get the dataframe without outliers

dataset_en=remove_outliers.reset_index()   # Resetting the index 
dataset_en=dataset_en.drop('index',axis=1)

corr=dataset_en.corr()
plt.figure(figsize = (10,10))
#sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns, annot= True,linewidths=.5)

#Correlation with output variable
cor_target = abs(corr['salary'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>.5]
relevant_features

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset_en, test_size=0.2, random_state=42)

X=train_set.drop('salary',axis=1)
y=train_set['salary']

X_test=test_set.drop('salary',axis=1)
y_test=test_set['salary']

from sklearn.linear_model import LassoCV

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)

imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

dataset_salpred=dataset_en.drop(['sl_no','ssc_b','hsc_b','degree_p'],axis=1)
train_set, test_set = train_test_split(dataset_salpred, test_size=0.2, random_state=42)


from xgboost import XGBRegressor
XGBreg=XGBRegressor(n_estimators = 100)
XGBreg.fit(X,y)
y_predict_xgb=XGBreg.predict(X_test)
from sklearn.metrics import r2_score
r2_xgbr=r2_score(y_test,y_predict_xgb)

from sklearn.ensemble import  ExtraTreesRegressor
ETree=ExtraTreesRegressor(n_estimators = 100)
ETree.fit(X,y)
y_predict=ETree.predict(X_test)
r2_ex=r2_score(y_test,y_predict)

from sklearn.tree import DecisionTreeClassifier 
clf=DecisionTreeClassifier(max_depth=3)
clf.fit(X,y)
y_pred_dt=clf.predict(X_test)
r2_dt=r2_score(y_test, y_pred_dt)

from xgboost import XGBClassifier
xgb_reg = XGBClassifier()
xgb_reg.fit(X, y)
y_pred_xgbc=clf.predict(X_test)
r2_xgbc=r2_score(y_test, y_pred_xgbc)

from sklearn.ensemble import RandomForestClassifier
forest_reg = RandomForestClassifier(max_depth=100, max_leaf_nodes=200, random_state=42)
forest_reg.fit(X,y)
y_predRF=forest_reg.predict(X_test)
r2_RF=r2_score(y_test, y_predRF)


