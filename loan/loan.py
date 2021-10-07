import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics

import seaborn as s


df=pd.read_csv('data.csv')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['pay_schedule']=le.fit_transform(df['pay_schedule'])


Q1 = df.quantile(0.05)
Q3 = df.quantile(0.80)
IQR = Q3 - Q1

index_outliers= df.loc[((df< (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)].index.tolist() #Index of the outliers 
outlier_values=df.loc[index_outliers]    #List of values of outliers
remove_outliers=df[~((df< (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]   # Removing outliers from dataframe and get the dataframe without outliers

df=remove_outliers.reset_index()
df=df.drop(['index','entry_id'],axis=1)   # Resetting the index

# X = mms.transform(X)


df.dropna(subset=['income'],inplace=True)

X = df.drop(labels='e_signed',axis=1)
Y= df['e_signed']
colname=X.columns
df = (X - X.mean()) / (X.max() - X.min())
df = pd.concat([df, Y], axis=1) 


plt.rcParams['figure.figsize']=(10,5)
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('e_signed',y='age',data=df, ax=ax1)
s.boxplot('e_signed',y='pay_schedule',data=df, ax=ax2)
s.boxplot('e_signed',y='home_owner',data=df, ax=ax3)
s.boxplot('e_signed',y='income',data=df, ax=ax4)
s.boxplot('e_signed',y='months_employed',data=df, ax=ax5)
f.tight_layout()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
s.boxplot('e_signed',y='years_employed',data=df, ax=ax2)
s.boxplot('e_signed',y='current_address_year',data=df, ax=ax1)
s.boxplot('e_signed',y='personal_account_m',data=df, ax=ax3)
s.boxplot('e_signed',y='personal_account_y',data=df, ax=ax4)
s.boxplot('e_signed',y='has_debt',data=df, ax=ax5)    
f.tight_layout()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
s.boxplot('e_signed',y='amount_requested',data=df, ax=ax2)
s.boxplot('e_signed',y='risk_score',data=df, ax=ax1)
s.boxplot('e_signed',y='risk_score_2',data=df, ax=ax3)
s.boxplot('e_signed',y='risk_score_3',data=df, ax=ax4)
s.boxplot('e_signed',y='risk_score_4',data=df, ax=ax5)    
f.tight_layout()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
s.boxplot('e_signed',y='risk_score_5',data=df, ax=ax2)
s.boxplot('e_signed',y='ext_quality_score',data=df, ax=ax1)
s.boxplot('e_signed',y='ext_quality_score_2',data=df, ax=ax3)
s.boxplot('e_signed',y='inquiries_last_month',data=df, ax=ax3)   
f.tight_layout()

