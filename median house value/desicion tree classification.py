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
import numpy as np
from sklearn.tree import DecisionTreeClassifier 

clf=DecisionTreeClassifier(max_depth=3)
clf.fit(X,y)

y_pred_dt=clf.predict(X)
dt_mse=mean_squared_error(y,y_pred_dt)
dt_rmse=np.sqrt(dt_mse)
r2_dt=r2_score(y, y_pred_dt)

