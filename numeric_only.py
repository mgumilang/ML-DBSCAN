#%%
import pandas as pd

#%%
df = pd.read_csv('CencusIncomeUndersampled.csv', header=None)
df1 = df.loc[:, [0,2,4,10,11,12]]

#%%
cols = list(df1.columns)
for col in cols:
    df1[col] = (df1[col] - df1[col].mean()) / (df1[col].std())
    
#%%
df1.to_csv('normalized_undersampled.csv', index=False)