---
date: 2021-06-05 12:58
title: "Pandas DL Code Snippet"
categories: Pandas DL
tags: Pandas DL
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Code Snippet

## Label Encoder
```python
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()

for f in col_enc:
    if total_df[f].dtype=='object':
        print(f)
        lbl = LabelEncoder()
        lbl.fit(list(total_df[f].values.astype('str')))
        total_df[f] = lbl.transform(list(total_df[f].values.astype('str')))
        total_df[f] = total_df[f].astype('category')

for col in col_enc:
    ce = total_df[col].value_counts()
    total_df[col+'_Freq'] = total_df[col].map(ce)
    #print(col)
```   

## Count Encoding with Value Counts Map
```python
for col in total_df.columns:
    ce = total_df[col].value_counts()
    total_df[col+'_Freq'] = total_df[col].map(ce)
    #print(col)
```  

## Missing Values Table
```python
def missing_values_table(df):# Function to calculate missing values by column# Funct 
    mis_val = df.isnull().sum() # Total missing values
    mis_val_pct = 100 * df.isnull().sum() / len(df)# Percentage of missing values
    mis_val_df = pd.concat([mis_val, mis_val_pct], axis=1)# Make a table with the results
    mis_val_df_cols = mis_val_df.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})# Rename the columns
    mis_val_df_cols = mis_val_df_cols[mis_val_df_cols.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)# Sort the table by percentage of missing descending
    print ("Dataframe has " + str(df.shape[1]) + " columns.\n" 
           "There are " + str(mis_val_df_cols.shape[0]) + " cols having missing values.")# Print some summary information
    return mis_val_df_cols # Return the dataframe with missing information
```  

## Basic Groupby Aggregation
```python
def base_agg(df, cols):
    temp_df = df.groupby(['location','date']).count().reset_index()
    temp_df = temp_df[['location','date']]
    for i in cols:
        temp = df.groupby(['location','date']).agg({'{}'.format(i):{'{}_MAX'.format(i):'max',
                                                                    '{}_MIN'.format(i):'min',
                                                                    '{}_MEAN'.format(i):'mean',
                                                                   '{}_FIRST'.format(i):'first',
                                                                   '{}_LAST'.format(i):'last',
                                                                   '{}_STD'.format(i):'std'}})
        temp.columns = temp.columns.droplevel(0)
        temp = temp.reset_index()
        temp['{}_MAX_MIN_diff'.format(i)] = temp['{}_MAX'.format(i)]-temp['{}_MIN'.format(i)]
        temp['{}_FIRST_LAST_diff'.format(i)] = temp['{}_FIRST'.format(i)]-temp['{}_LAST'.format(i)]
        
        temp_df = temp_df.merge(temp, on=['location','date'], how='left')
    return temp_df
```  
```python
SEASON_S1 = SEASON.groupby(['batter_id','year']).agg({'avg':'mean','G':'sum','AB':'sum','R':'sum','H':'sum','2B':'sum','3B':'sum','HR':'sum','TB':'sum','RBI':'sum','SB':'sum','CS':'sum','BB':'sum','HBP':'sum','SO':'sum','GDP':'sum',
                                                     '1B':'sum','SLG':'mean','OBP':'mean','E':'sum','OPS':'mean'})
SEASON_S1 = SEASON_S1.unstack()
SEASON_S1.columns = SEASON_S1.columns.map('{0[1]}|{0[0]}'.format)
SEASON_S1.head(6)
```  
```python
agg = {"transacted_date":{"last_trans":"max"},
      "END_DATE":{"END_DATE":"max"}}
DF_no = DF.groupby(['store_id']).agg(agg)
DF_no.columns = DF_no.columns.droplevel(0) # 순서 중요 agg 하고 droplevel 하고 reset index
DF_no = DF_no.reset_index()
DF_no.head()
```



## Filter

```python
test_df.filter(regex=("station.*")).columns.tolist()
```   
or operator:  
```python
test_df.filter(regex=("bus_route_id|station|in_out.*")).columns.tolist()
```  

## str.split
expand:  
```python
aa['code','name'] = aa['station_id'].str.split("_", expand=True)
```  

## Isna & Imputation
show df with NaN:  
```python
df[df.isna().any(axis=1)]
```  
Imputation with loc:  
```python
df.loc[df['Rating'].isna(),'Rating'] = 0
```  

## Groupby & Transform
```python
for df in [train_df, test_df]:
    print('Started')
    for col in rideandoff:
        df['uniq_bus_route_{}_sum'.format(col)] = df.groupby('uniq_bus_route')[col].transform('sum')
        df['uniq_bus_route_{}_min'.format(col)] = df.groupby('uniq_bus_route')[col].transform('min')
        df['uniq_bus_route_{}_max'.format(col)] = df.groupby('uniq_bus_route')[col].transform('max')
        df['uniq_bus_route_{}_count'.format(col)] = df.groupby('uniq_bus_route')[col].transform('count')
        df['uniq_bus_route_{}_std'.format(col)] = df.groupby('uniq_bus_route')[col].transform('std')
        print(col)
```  

## Shift
```python
rideandoff = test_df.filter(regex=("ride.*")).columns.tolist()+test_df.filter(regex=("takeoff.*")).columns.tolist()

shifted_m1 = train_df.groupby("uid_1")[rideandoff].shift(-1)

shifted_m1.columns = 'lag_' + shifted_m1.columns
```  

## Fill NaN
```python
test1_g[null_list] = test1_g[null_list].fillna(0)
```  

## Sum
row-wise with specific columns:  
```python
df['row-wise-sum'] = df[col_list].sum(axis=1, skipna=True)
```  

## Value Counts unstack
```python
day_list = ['Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday']
train_df_new12 = train_df.groupby(['store_id','year','month'])['dayofweek'].value_counts(dropna=False, normalize=True).unstack()
# -------CAUTION COLUMNS NAMES ARE ORDERED BY ALPHABETICAL.----------
train_df_new12.columns = ['day_ratio'+i for i in day_list]
train_df_new12 = train_df_new12.reset_index()
```  

## Select_dtypes
```python
total_df.select_dtypes(include=[object,'category']).columns
```  

## To Binary
```python
df['flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
```  

## Frequency Encoding (qcut)
```python
for i in list(train.columns.values[2:202]):
    encoding_i = train.groupby(i).size()
    encoding_i = encoding_i/len(train)
    train[i+'_qcut_FQE'] = pd.qcut(train[i], 20, labels=np.arange(20)).astype(int)
    train[i+'_qcut_FQE'] = train[i].map(encoding_i)
```  

## Remove Duplicates columns
```python
df = df.loc[:,~df.columns.duplicated()]
```   

## Replace inf to zero
```python
df.replace([np.inf, -np.inf], 0, inplace=True)
```  

## Get column(index) name based on row-wise min
```python
train_df['nearest_obsv'] = train_df[obsv_list].idxmin(axis=1)
```  

## Consecutive time difference
```python
DF_DC['Days'] = (DF_DC.groupby('store_id', group_keys=False)
                .apply(lambda g: g['transacted_date'].diff().replace(0, np.nan).ffill()))
```  

## Select rows with condition
based on value with a specific column:  
```python
DF_no[DF_no['store_id'].isin(closed_ids)]
```  

## ffill Vs. bfill
```python
import pandas as pd
df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, 5],
                [np.nan, 3, np.nan, 4]],
                columns=list('ABCD'))
print(df)
print(df.fillna(method='ffill'))
# fill NaN going forward
print(df.fillna(method='bfill'))
# fill NaN going backward
```
```
     A    B   C  D
0  NaN  2.0 NaN  0
1  3.0  4.0 NaN  1
2  NaN  NaN NaN  5
3  NaN  3.0 NaN  4
     A    B   C  D
0  NaN  2.0 NaN  0
1  3.0  4.0 NaN  1
2  3.0  4.0 NaN  5
3  3.0  3.0 NaN  4
     A    B   C  D
0  3.0  2.0 NaN  0
1  3.0  4.0 NaN  1
2  NaN  3.0 NaN  5
3  NaN  3.0 NaN  4
```  

## Pandas float_format
```python
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.float_format', None) #원래상태로
```  

## Week number with codition
```python
def conditions(x):
    if x < 8:
        return 1
    elif x < 15:
        return 2
    elif x < 22:
        return 3
    else:
        return 4

func = np.vectorize(conditions)
week_num = func(train_payment["day"])
train_payment["week_num"] = week_num
```  

