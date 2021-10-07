import pandas as pd

df=pd.read_excel(r'Concrete_Data.xls')

df2=df.describe()

df1=df.isin([0]).sum()

median=df.median()

#What are co-relation techinques (df.corr)

corr=df.corr()
#Which column to drop from data set

#draw a co-relation heat map using seaborn package


import seaborn as sns

#what is pearsons corr

#replace 1 and 4 th row with mean of respective columns

features=df.shape[1]

for col in range(0,features):
    mean4=df[df.columns[col]].mean()
    df[df.columns[col]].fillna(mean4,inplace=True)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

index_outliers= df.loc[((df< (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)].index.tolist() #Index of the outliers 
outlier_values=df.loc[index_outliers]    #List of values of outliers
remove_outliers=df[~((df< (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

remove_outliers.reset_index(inplace=True)
clean_df=remove_outliers.drop('index',axis=1)
clean_dff=pd.get_dummies(clean_df)
#df.plot.scatter(x='total_bedrooms',y='median_house_value',title='scatter plot') 

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)

#coff of determination r2











