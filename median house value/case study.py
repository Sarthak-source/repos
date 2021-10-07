import pandas as pd

df=pd.read_csv(r'case_study_preprocessing_L1.csv')

df.quantile(0.05)

for i in range(0,8):
    df=df[(df[df.columns[i]]>=df[df.columns[i]].quantile(0.05)) & (df[df.columns[i]]<=df[df.columns[i]].quantile(0.95))]
    
df_final=pd.get_dummies(df)

df.plot.scatter(x='total_bedrooms',y='median_house_value',title='scatter plot')


df2=df.insert(column=0,value=df.index,loc=0)
df.reset_index()

#df.plot.scatter(x=df.index,y='total_bedrooms',title='statter plot 2')

#df.boxplot(by=['total_bedrooms'])

#types of encoding 

#label,one hot encode

#assignment (diff between merge ,concatinate)

