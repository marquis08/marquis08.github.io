---
date: 2021-06-06 13:42
title: "ML basics - MSE & MAE"
categories: DevCourse2 MathJax DevCourse2_ML_Basics
tags: DevCourse2 MathJax DevCourse2_ML_Basics
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Dataset
{%raw%}
|    | Restaurant   | Location                            | Cuisines                               |   AverageCost |   MinimumOrder |   Rating |   Votes |   Reviews |   DeliveryTime |
|---:|:-------------|:------------------------------------|:---------------------------------------|--------------:|---------------:|---------:|--------:|----------:|---------------:|
|  0 | ID6321       | FTI College, Law College Road, Pune | Fast Food, Rolls, Burger, Salad, Wraps |           200 |             50 |      3.5 |      12 |         4 |             30 |
|  1 | ID2882       | Sector 3, Marathalli                | Ice Cream, Desserts                    |           100 |             50 |      3.5 |      11 |         4 |             30 |
|  2 | ID1595       | Mumbai Central                      | Italian, Street Food, Fast Food        |           150 |             50 |      3.6 |      99 |        30 |             65 |
|  3 | ID5929       | Sector 1, Noida                     | Mughlai, North Indian, Chinese         |           250 |             99 |      3.7 |     176 |        95 |             30 |
|  4 | ID6123       | Rmz Centennial, I Gate, Whitefield  | Cafe, Beverages                        |           200 |             99 |      3.2 |     521 |       235 |             65 |
{%endraw%}  

# Preprocessing
## OHE
```python
df = pd.concat([df, df['Cuisines'].str.get_dummies(sep=',').add_prefix('Cuisines_')],axis=1)
df = pd.concat([df, df['Location'].str.get_dummies(sep=',').add_prefix('Location_')],axis=1)
```  
## Count Encoding
```python
# Count Encoding
ce = df['Restaurant'].value_counts()
df['Restaurant'+'_Freq'] = df['Restaurant'].map(ce)
df.reset_index(inplace=True,drop=True)
```
## Categorization
```python
cate_cols = ['Restaurant','Rating','Location','Cuisines','AverageCost']
for c in cate_cols:
    df[c] = df[c].astype('category')
```  
## Remove white space in column names
```python
import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
```

# Split Train & Test
Stratified Holdout 0.2:  
```python
train, test, _, _ = train_test_split(df, df['DeliveryTime'], test_size = 0.2, shuffle=True, stratify=df['DeliveryTime'])
```  

# Run Model
## MAE
```python
metric = 'mae'
params = {
                 'objective': metric, # loss
                 'max_depth': -1,
                 'learning_rate': 0.01,
                 "boosting": "gbdt",
                 "metric": metric,
                 "verbosity": -1,
                 "nthread": -1,
                 "random_state": 42,
                'n_estimators': 20000, # num_iterations 
                }

preds, oof, cv_mean, cv_std, best_feats = LGB_STRATAKFOLD_REG(5, X_tr, X_te, metric, train_Y, params)

result = preds < test['DeliveryTime'].values
UnderPrediction = sum(result)/len(result)

print("\n"+'#'*40,"\n## SKF5 - {}:\t\t\t {:0.4f}".format(metric.upper(),cv_mean))
print("## SKF5 - UnderPrediction:\t {:0.4f}".format(UnderPrediction)+"\n"+'#'*40)
```
```
                              Feature  importance  Feature Rank
172                             Votes    25147.60           1.0
171                           Reviews    18196.20           2.0
170                   Restaurant_Freq     9977.80           3.0
168                            Rating     8791.40           4.0
0                         AverageCost     4370.40           5.0
######################################## 
## SKF5 - MAE:			    5.3994
## SKF5 - UnderPrediction:	    0.4403
########################################
```  
## MSE
```python
metric = 'mse'
params = {
                 'objective': metric, # loss
                 'max_depth': -1,
                 'learning_rate': 0.01,
                 "boosting": "gbdt",
                 "metric": metric,
                 "verbosity": -1,
                 "nthread": -1,
                 "random_state": 42,
                'n_estimators': 20000, # num_iterations 
                }

preds, oof, cv_mean, cv_std, best_feats = LGB_STRATAKFOLD_REG(5, X_tr, X_te, metric, train_Y, params)

result = preds < test['DeliveryTime'].values
UnderPrediction = sum(result)/len(result)

print("\n"+'#'*40,"\n## SKF5 - {}:\t\t\t {:0.4f}".format(metric.upper(),cv_mean))
print("## SKF5 - UnderPrediction:\t {:0.4f}".format(UnderPrediction)+"\n"+'#'*40)
```
```
                  Feature  importance  Feature Rank
172                 Votes    29995.80           1.0
171               Reviews    26825.80           2.0
170       Restaurant_Freq     9766.80           3.0
168                Rating     7717.20           4.0
0             AverageCost     4006.80           5.0
######################################## 
## SKF5 - MSE:			97.0386
## SKF5 - UnderPrediction:	0.3335
########################################
```  

# MAE Vs. MSE and RMSE
\\[ MAE = \frac{1}{N}\sum_{i=1}^{N}|y_{i}-\hat{y}| \\]  
\\[ MSE = \frac{1}{N}\sum_{i=1}^{N}(y_{i}-\hat{y})^{2} \\]  
\\[ RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_{i}-\hat{y})^{2}} \\]  

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_preds)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mean_squared_error(y_test, y_preds)
```  


{%raw%}
|    | Prediction | GroundTruth | Diff      | Abs Diff   |
|---:|:-----------|:------------|:----------|-----------:|
|  0 | 400,000     | 420,000      |  20,000    | 20,000      |
|  1 | 250,000     | 234,000      |  -16,000   | 16,000      |
|  2 | 800,000     | 760,400      |  -39,600   | 39,600      |
|   |      |       |  MAE   | 25,200        |
{%endraw%}  

Interpretation:  
This model produce avg 25200 wrong prediction, intuitively.  

{%raw%}
|    | Prediction | GroundTruth | Diff      | Abs Diff   |
|---:|:-----------|:------------|:----------|-----------:|
|  0 | 400,000     | 420,000      |  20,000    | 20,000      |
|  1 | 250,000     | 234,000      |  -16,000   | 16,000      |
|  2 | 800,000     | 760,400      |  -39,600   | 39,600      |
|   |      |       |  MAE   | 25,200        |
|   |      |       |  RMSE   | 27,228        |
{%endraw%}  

Can't say model wrongly predict result with avg 27,228, since we use square and square root. 

{%raw%}
|    | Prediction | GroundTruth | Diff      | Abs Diff   |
|---:|:-----------|:------------|:----------|-----------:|
|  0 | 400,000     | 420,000      |  20,000    | 20,000    |
|  1 | 250,000     | 234,000      |  -16,000   | 16,000    |
|  2 | 800,000     | 760,400      |  -39,600   | 39,600    |
|  3 | 29,342,000  | 34,234,400   |  4,892,000 |  4,892,000|
|    |             |              |  MAE(before outlier)  | 25,200        |
|    |             |              |  RMSE(before outlier) | 27,228        |
|    |             |              |  MSE(before outlier) | 7,413,866,666  |
|    |             |              |  MAE(after outlier)   | 1,241,900     |
|    |             |              |  RMSE(after outlier)  | 2,446,144     |
|    |             |              |  MSE(after outlier) | 5,984,450,480,000  |
{%endraw%}  


## MAE
MAE is more robust to data with outliers.  

## MSE & RMSE
MSE penalize the large prediction errors with square.  
This makes more sensitive to the outliers.  


> Since MSE squares the error($$e$$), the value of error($$e$$) increases a lot if $$e > 1$$. This will make the model with MSE loss give more weight to outliers than a model with MAE loss. The model with RMSE as loss will be adjusted to minimize outliers at the expense of other common examples, which will reduce its overall performance.  

MAE loss is useful if the training data is corrupted with outliers.  

Intuitively, we can think about it like this: If we only had to give one predction for all the observations that try to minimize MSE, then that prediction should be the **mean** of all target values. But if we try to minimize MAE, that prediction would be the **median** of all observations.  

## MSE predicts less on UnderPrediction. Why?  
```
#######################################
## SKF5 - MAE:			5.3994
## SKF5 - UnderPrediction:	0.4403

## SKF5 - MSE:			97.0386
## SKF5 - UnderPrediction:	0.3335
#######################################
```  
Since it squares the error, MSE predicts more OverPrediction then MAE.  

# Summary
Under the circumstance of few outliers, if you want the model to minimize UnderPrediction, use MSE.  
Under the circumstance of many outliers, use MAE.  
* UnderPrediction: predicted delivery time < true delivery time  

If minimize UnderPrediction is your goal, use mse for loss function.  
If lots of outliers in dataset, use mae for loss function.  

- **MAE** fits on the basis of the median.  
- **MSE** fits on the basis of the mean.  
- **RMSE** solves the problem of squaring the units. 




# Appendix
## pandas to md table
install tabulate:  
```
pip install tabulate
```  
```python
print(df.head().to_markdown())
```  


## Reference
> df to md table: <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_markdown.html>  
> MSE MAE: <https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e>  
> MSE MAE: <https://data101.oopy.io/mae-vs-rmse>  
> MSE MAE: <https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0>  
> MSE MAE: <http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/>  
> MSE MAE: <https://www.kaggle.com/c/home-data-for-ml-course/discussion/143364>  


