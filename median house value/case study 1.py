import pandas as pd

df=pd.read_csv(r'case_study_preprocessing_L1.csv')


#to find the data types of rows and coloumn
df.dtypes

#information about dataframe
df.info()

#stats of dataframe
df.describe()

#number of rows missing values
df.isnull().sum(axis=1)

#get first 10 rows 
df.head(10)

#get the last 7 rows
df.tail(7)

#index of dataframe
df.index

#values of columns
df.columns

#get row total_bedrooms
df1=df['total_bedrooms']

#get fourth row
df2=df.iloc[:,4]

df3=df[df.columns[4]]

#transpose data frame 
df4=df.transpose()

#find null values
df5=df4.isnull()

#diff between lock and ilock(assignment)

#mean of colomn
mean=df3.mean()

#fill missing value
df['total_bedrooms'].fillna(mean,inplace=True)

#Drop missing values
df.dropna(inplace=True)

#assignment to find function which find index of missing value

#df.to_csv('nomissingvalue.csv')

#catogosie df
df['ocean_proximity'].value_counts()

#save description to csv
dis=df.describe()
dis.to_csv('describe.csv')

#df.quantile(5)




