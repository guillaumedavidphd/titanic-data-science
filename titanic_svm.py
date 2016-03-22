# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:46:30 2016

@author: guillaumedavidphd
"""

#%% Imports
import pandas as pd
import numpy as np

#%% Read and preprocess data
df_titanic = pd.read_csv("train.csv", index_col="PassengerId")
# df_titanic.drop(["Ticket", "Cabin", "Name"], axis=1, inplace=True)

df_titanic.Age.fillna(df_titanic.Age.median(), inplace=True)

df_titanic.loc[df_titanic["Sex"] == "male", "Sex"] = 0
df_titanic.loc[df_titanic["Sex"] == "female", "Sex"] = 1

df_titanic.Embarked.fillna("S", inplace=True)
df_titanic.loc[df_titanic["Embarked"] == "S", "Embarked"] = 0
df_titanic.loc[df_titanic["Embarked"] == "C", "Embarked"] = 1
df_titanic.loc[df_titanic["Embarked"] == "Q", "Embarked"] = 2

