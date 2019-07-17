# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:34:33 2019

@author: Big data
"""

import xgboost as xgb
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
def AVM(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    Z = np.where(abs((y_pred - y_test)/y_test) <= 0.3, 1, 0)
    hit_rate = Z.sum()/len(Z)
    MAPE = (abs((y_pred - y_test)/y_test)).sum()/len(Z)
    score = hit_rate*10000 + (1-MAPE)
    return print(hit_rate)

df_fill = pd.read_csv('final_620_US.csv',encoding="utf-8")
df_fill = df_fill[df_fill["Domestic_box_office"].notna() ]
df_fill = df_fill[df_fill["Domestic_box_office"]!='$0']
df_fill = df_fill.drop(df_fill[df_fill["name"].duplicated()].index)
df_fill = df_fill.drop(['Unnamed: 0',"opening_weekend","gross_tw","release_date_USA","Awards"
              ],axis=1)
cleaned = df_fill.set_index('name').Genre.str.split(',', expand=True).stack()
cleaned = cleaned.apply(lambda x:x.replace(" ",""))
genre_enc = pd.get_dummies(cleaned, prefix='g').groupby(level=0,sort=False).sum()

cleaned = df_fill.set_index('name').Language.str.split(',', expand=True).stack()
cleaned = cleaned.apply(lambda x:x.replace(" ",""))
drop_lang = (cleaned.value_counts()[cleaned.value_counts()<=6]).index.tolist()
for l in range(len(cleaned)):
    if cleaned.iloc[l] in drop_lang:
        cleaned.iloc[l] = "Other_language"
language_enc = pd.get_dummies(cleaned, prefix='L').groupby(level=0,sort=False).sum()

cleaned = df_fill.set_index('name').Country.str.split(',', expand=True).stack()
cleaned = cleaned.apply(lambda x:x.replace(" ",""))
country_enc = pd.get_dummies(cleaned, prefix='L').groupby(level=0,sort=False).sum()
df_fill.loc[25,"budget_in_USD"] = 7000000 
df_fill["runtime"] = df_fill["runtime"].str.replace("min","")
df_fill["imdbVotes"]=df_fill["imdbVotes"].str.replace(",","")
for e in df_fill.columns:
    if "movie" or "Actor" in e:
        df_fill[e] = df_fill[e].replace("error","0")
df_fill["Domestic_box_office"]=df_fill["Domestic_box_office"].str.replace("$","").str.replace(",","")
cla_enc = pd.get_dummies(df_fill["classification"])
df_fill = pd.concat([df_fill.reset_index(drop=True),genre_enc.reset_index(drop=True),
                language_enc.reset_index(drop=True),country_enc.reset_index(drop=True),
                cla_enc.reset_index(drop=True),],
                axis=1)


df_fill = df_fill.drop(["year","name","Genre","Director","Language","Country","Production","classification","movie_8_before",
              "movie_7_before","movie_6_before","movie_5_before","movie_4_before","movie_3_before"
              ,"Actor_8_before","Actor_7_before","Actor_6_before"
              ,"Actor_5_before","Actor_4_before","Actor_3_before","Domestic_box_office"
              ],axis=1)
df_train = df_fill[df_fill["budget_in_USD"].notna()]
df_test = df_fill[df_fill["budget_in_USD"].isna()]
df_test = df_fill.drop("budget_in_USD",axis=1)
y= df_train.pop("budget_in_USD")
for i in df_train.columns:
    print(i)
    df_train[i] =pd.to_numeric(df_train[i], downcast='float')
    
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.1, random_state=42)
y_train = np.log(y_train)
model = xgb.XGBRegressor(colsample_bytree=0.9, gamma=0.1, 
                                max_depth=8, 
                               min_child_weight=0.05, n_estimators=300,
                               subsample=0.6, silent=1,nthread =8,tree_method="hist",
                               random_state =7)
model.fit(X_train, y_train)
#scores = cross_val_score(model, X_train, y_train, cv=5)
#print(scores)
y_pred = model.predict(X_test)
y_pred = np.exp(y_pred)
AVM(y_test,y_pred)
from sklearn.metrics import  r2_score
print(r2_score(y_test,y_pred))