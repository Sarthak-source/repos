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


################### Removing outiers using IQR ####################################################
Q1 = df1.quantile(0.25)
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
