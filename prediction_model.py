# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:14:37 2019
df.columns[df.isna().any()].tolist()
def AVM(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    Z = np.where(abs((y_pred - y_test)/y_test) <= 0.1, 1, 0)
    hit_rate = Z.sum()/len(Z)
    MAPE = (abs((y_pred - y_test)/y_test)).sum()/len(Z)
    score = hit_rate*10000 + (1-MAPE)
    return print(score)
    
    for i, s in df_small.iterrows():
        #print(s["regexname"], s["Movie_name"])
        filled[s["regexname"]] = [s["Movie_name"],s["Domestic_box_office"],s["International_box_office"],
                                  s["Worldwide_box_office"],s["Theater_num"]]
    
    for i, s in df_big.iterrows():
        
        if s["regexname"] in filled:
            df_big.loc[i, "Movie_name"] = filled[s["regexname"]][0]
            df_big.loc[i, "Domestic_box_office"] = filled[s["regexname"]][1]
            df_big.loc[i, "International_box_office"] = filled[s["regexname"]][2]
            df_big.loc[i, "Worldwide_box_office"] = filled[s["regexname"]][3]
            df_big.loc[i, "Theater_num"] = filled[s["regexname"]][4]
@author: Big data
"""
import seaborn as sns
import re
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


df = pd.read_csv('628_1.csv',encoding="utf-8")
df = df.drop(["Unnamed: 0","id","opening_weekend","gross_g","gross_tw","gross_cn","gross_hk","gross_my"
              ,"gross_sg","gross_xndom","Actors","ChineseName","TaiwanRelease","Domestic_box_office",
              "International_box_office","Worldwide_box_office","Cmovie_8_before","Cmovie_7_before",
              "Cmovie_6_before","Cmovie_5_before","Cmovie_4_before","Cmovie_3_before","Cmovie_2_before",
              "Cmovie_1_before","Cmovie_0_before","Cmovie_1_after","Cmovie_2_after","Cmovie_3_after",
              "Cmovie_4_after","Cmovie_5_after","Cmovie_6_after","Cmovie_7_after","Cmovie_8_after"],axis=1)
df.loc[25,"budget_in_USD"] = 7000000 
df['Runtime']=df["runtime"]
df["Domestic_box_office"] = df["gross_na"]
df= df[df["Domestic_box_office"].notna() & df["Domestic_box_office"].notna()]
df=df[df["Domestic_box_office"]!='$0'].reset_index(drop=True)


regex_pat = re.compile(r"[^a-zA-Z0-9]+", flags=re.IGNORECASE)
df_company = pd.read_csv("company_detail.csv")
df_company["domestic"]=df_company["domestic"].str.replace("$","").str.replace(",","")
df_company["domestic"]=df_company["domestic"].astype("int64")
df_company["company"] = df_company["company"].str.replace(regex_pat,"").str.lower()
df_company["avg"] = df_company["domestic"]/df_company["movie_num"]

company_dict = {}
for i, s in df_company.iterrows():
    company_dict[s["company"]] = s["avg"]
company_dict["theweinsteincompany"] = 30985651.243902437
company_dict["mgm"] = 57700692.79518072
company_dict["paramountstudios"] = 89797383.81300813
company_dict["theweinsteinco"] = 30985651.243902437
company_dict["a24films"] = 7931751.708333333
company_dict["theweinsteinco"] = 30985651.243902437
company_dict["theweinsteinco"] = 30985651.243902437
company_dict["universalstudios"] = 77798237.0
company_dict["sonyscreengems"] = 22962059.904761903
company_dict["a24anddirectv"] = 7931751.708333333

company_dict["buenavistapictures"] = 38871828.0
company_dict["sonycolumbiapictures"] = 22962059.904761903
company_dict["sonypicturesentertainment"] = 22962059.904761903
company_dict["sonyclassics"] = 3170505.722222222
company_dict["ifc"] = 241840.0
company_dict["firstlookpictures"] = 22527888.0
company_dict["firstlookstudios"] = 3170505.722222222
company_dict["sonyclassics"] = 3170505.722222222
company_dict["waltdisneystudios"] = 13241726.0
company_dict["screengemssonypictures"] = 3170505.722222222
company_dict["sony"] = 3170505.722222222
company_dict["dreamworksdistributionllc"] = 13241726.0
company_dict["dreamworksparamount"] = 81498905.15853658
company_dict["waltdisneystudios"] = 117009673.77419356
company_dict["warnerbrospicturesnewlinecinema"] = 81913421.88990825
key = list(company_dict.keys())
    
regex_pat = re.compile(r"[^a-zA-Z0-9]+", flags=re.IGNORECASE)
df["Production"] = df["Production"].str.replace(regex_pat,"").str.lower()
df["Production"] = df["Production"].fillna("Other_studio")

#df_company = df_company.drop(["movie_num","domestic","international"],axis=1)
#df_company.to_csv("company_detail.csv",index=False)
        



count = 0
for i, s in df.iterrows():
    temp = []
    for x in key:
        if s["Production"].find(x) != -1:
            temp.append(x)

            #if s["Production"].rfind(x) == 0:
    for h in temp:
        if s["Production"].startswith(h) == 1 and ((len(h)>len(s["Production"])-len(h)) or len(s["Production"])-len(h)==0):
            df.loc[i,"Production"]= company_dict[h]
            print(s["Production"],'  ',h)
            count+=1
            print(count)
            break
    if df.loc[i,"Production"] != company_dict[h]:
        for x in key:
            if x.find(s["Production"]) != -1:
                temp.append(x)
        for h in temp:
                if h.startswith(s["Production"]) == 1 and ((len(s["Production"])>len(h)-len(s["Production"])) or len(s["Production"])-len(h)==0):
                    df.loc[i,"Production"]= company_dict[h] 
                    print(s["Production"],'  ',h)
                    count+=1
                    print(count)
                    break
                
df = df[df["budget_in_USD"].notna()]
df = df.reset_index(drop=True)
df["Domestic_box_office"] = df["Domestic_box_office"].str.replace("$","").str.replace(",","")
df["Domestic_box_office"] = df["Domestic_box_office"].astype("float")
df["budget_in_USD"] = df["budget_in_USD"].astype("float")
df = df[df["Domestic_box_office"]-df["budget_in_USD"]>0]
'''
df = df[df["budget_in_USD"].notna()]
df = df.reset_index(drop=True)
df["Domestic_box_office"] = df["Domestic_box_office"].str.replace("$","").str.replace(",","")
df["Domestic_box_office"] = df["Domestic_box_office"].astype("float")
df["budget_in_USD"] = df["budget_in_USD"].str.replace("$","").str.replace(",","")
df["budget_in_USD"] = df["budget_in_USD"].astype("float")
df = df[df["Domestic_box_office"]-df["budget_in_USD"]>0]         
'''
df["Production"] = df["Production"].astype("str")        
df["Production"] = df["Production"].apply(lambda x:np.nan if x.isalnum() or x =="Other_studio" else x)  
df["Production"] = df["Production"].fillna(df["Production"].median())
df["Production"] = df["Production"].astype("float")   






df = df.drop(df[df["name"].duplicated()].index)
cleaned = df.set_index('name').Genre.str.split(',', expand=True).stack()
cleaned = cleaned.apply(lambda x:x.replace(" ",""))
genre_enc = pd.get_dummies(cleaned, prefix='g').groupby(level=0,sort=False).sum()

cleaned = df.set_index('name').Language.str.split(',', expand=True).stack()
cleaned = cleaned.apply(lambda x:x.replace(" ",""))
drop_lang = (cleaned.value_counts()[cleaned.value_counts()<=5]).index.tolist()
for l in range(len(cleaned)):
    if cleaned.iloc[l] in drop_lang:
        cleaned.iloc[l] = "Other_language"
language_enc = pd.get_dummies(cleaned, prefix='L').groupby(level=0,sort=False).sum()

cleaned = df.set_index('name').Country.str.split(',', expand=True).stack()
cleaned = cleaned.apply(lambda x:x.replace(" ",""))
country_enc = pd.get_dummies(cleaned, prefix='L').groupby(level=0,sort=False).sum()
'''
drop_pro = (df["Production"].value_counts()[df["Production"].value_counts()<=10]).index.tolist()
for p in range(len(df["Production"])):
    if df["Production"].iloc[p] in drop_pro:
        df["Production"].iloc[p] = "Other_studio"
Production_enc = pd.get_dummies(df["Production"], prefix='L')
'''


df["Runtime"] = df["Runtime"].str.replace("min","")
df["imdbVotes"]=df["imdbVotes"].str.replace(",","")
for e in df.columns:
    if "movie" or "Actor" in e:
        df[e] = df[e].replace("error","0")

cla_enc = pd.get_dummies(df["classification"])
df = df.drop(["year","name","Genre","Director","release_date_USA","Language","Country","classification","movie_8_before",
              "movie_7_before","movie_6_before","movie_5_before","movie_4_before","movie_3_before"
              ,"Actor_8_before","Actor_7_before","Actor_6_before"
              ,"Actor_5_before","Actor_4_before","Actor_3_before","gross_na",
              "release_date","Writer","Awards","movie_1_after","movie_2_after",
              "movie_3_after","movie_4_after","movie_5_after","movie_6_after","movie_7_after",
              "movie_8_after","Actor_1_after","Actor_2_after","Actor_3_after","Actor_4_after",
              "Actor_5_after","Actor_6_after","Actor_7_after","Actor_8_after","runtime"],axis=1)
#df.to_csv("fillna.csv",index=False)
#y = df.pop("gross_na")
#y = df.pop("Domestic_box_office")
y = df.pop("Domestic_box_office")
for i in df.columns:
    print(i)
    df[i] =pd.to_numeric(df[i], downcast='float')
#df["budget_in_USD"] = np.log(df["budget_in_USD"]) 
    
y = y.astype(int)
#df["Production"] = np.log(df["Production"])   
df["Theater_num"] = np.log(df["Theater_num"]) 

df = pd.concat([df.reset_index(drop=True),genre_enc.reset_index(drop=True),
                country_enc.reset_index(drop=True),
                cla_enc.reset_index(drop=True)
                ],
                axis=1) 

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.05, random_state=42)
y_train = np.log(y_train)
model = xgb.XGBRegressor(colsample_bytree=0.8, gamma=0.1, 
                                max_depth=8, 
                               min_child_weight=0.8, n_estimators=900,
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
from xgboost import plot_importance
xgb.plot_importance(model,max_num_features=15)

aaa = model.get_booster()
aaa.dump_model("dump_raw.txt")
import joblib
joblib.dump(model, "modeltest")
import pickle
pickle.dump(model, open("gain.pickle.dat", "wb"))


'''
model = xgb.XGBRegressor(colsample_bytree=0.8, gamma=0.1, 
                                max_depth=8, 
                               min_child_weight=0.8, n_estimators=800,
                               subsample=0.7, silent=1,nthread =8,tree_method="hist",
                               random_state =7)
model.fit(X_train, y_train)
#scores = cross_val_score(model, X_train, y_train, cv=5)
#print(scores)
y_pred = model.predict(X_test)
y_pred = np.exp(y_pred)
AVM(y_test,y_pred)
0.6666666666666666
'''