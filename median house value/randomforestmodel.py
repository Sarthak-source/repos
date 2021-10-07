import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('8_Nov_case1_clean_data_.csv')
#df =df.reset_index(drop = True)

noice=[0]

df =df.drop(df.columns[noice],axis=1)

cordata=df.corr()

plt.figure(figsize = (16,10))

sns.heatmap(cordata,xticklabels=cordata.columns,yticklabels=cordata.columns, annot= True,linewidths=.5)





################# ML part #################################
################# split_train_test#################

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

X=train_set.drop('median_house_value',axis=1)
y=train_set['median_house_value']

X_test=test_set.drop('median_house_value',axis=1)
y_test=test_set['median_house_value']

from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
# Train the model on training data
rf.fit(X, y)

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X,y)

best=rf_random.best_params_

rf_best=RandomForestRegressor(**best)
rf_best.fit(X,y)


y_pred_rf=rf_best.predict(X)
rf_mse=mean_squared_error(y,y_pred_rf)
rf_rmse=np.sqrt(rf_mse)
r2_rf=r2_score(y, y_pred_rf)

########### test rf model ##################

y_pred_rf_t=rf_best.predict(X_test)
rf_mse_t=mean_squared_error(y_test,y_pred_rf_t)
rf_rmse_t=np.sqrt(rf_mse_t)
r2_rf_t=r2_score(y_test, y_pred_rf_t)


