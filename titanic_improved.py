#!/usr/bin/env python
"""This script makes prediction on Titanic survivors data."""

import pandas as pa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import re
import operator
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

df_titanic = pa.read_csv("train.csv", index_col="PassengerId")
# df_titanic.drop(["Ticket", "Cabin", "Name"], axis=1, inplace=True)

df_titanic.Age.fillna(df_titanic.Age.median(), inplace=True)

df_titanic.loc[df_titanic["Sex"] == "male", "Sex"] = 0
df_titanic.loc[df_titanic["Sex"] == "female", "Sex"] = 1

df_titanic.Embarked.fillna("S", inplace=True)
df_titanic.loc[df_titanic["Embarked"] == "S", "Embarked"] = 0
df_titanic.loc[df_titanic["Embarked"] == "C", "Embarked"] = 1
df_titanic.loc[df_titanic["Embarked"] == "Q", "Embarked"] = 2

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = RandomForestClassifier(random_state=1,
                             n_estimators=150,
                             min_samples_split=4,
                             min_samples_leaf=2)

scores = cross_validation.cross_val_score(alg,
                                          df_titanic[predictors],
                                          df_titanic.Survived,
                                          cv=3)

print(scores.mean())

df_titanic["FamilySize"] = df_titanic.SibSp + df_titanic.Parch
df_titanic["NameLength"] = df_titanic.Name.apply(lambda x: len(x))


def get_title(name):
    """Get title of person (Mr., Ms., etc)."""
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

titles = df_titanic.Name.apply(get_title)
print(titles.value_counts)

title_mapping = {"Mr": 1,
                 "Miss": 2,
                 "Mrs": 3,
                 "Master": 4,
                 "Dr": 5,
                 "Rev": 6,
                 "Major": 7,
                 "Col": 7,
                 "Mlle": 8,
                 "Mme": 8,
                 "Don": 9,
                 "Lady": 10,
                 "Countess": 10,
                 "Jonkheer": 10,
                 "Sir": 9,
                 "Capt": 7,
                 "Ms": 2}

for k, v in title_mapping.items():
    titles[titles == k] = v

df_titanic["Title"] = titles

family_id_mapping = {}


def get_family_id(row):
    """Get unique family ID."""
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(),
                          key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

family_ids = df_titanic.apply(get_family_id, axis=1)

family_ids[df_titanic["FamilySize"] < 3] = -1

df_titanic["FamilyId"] = family_ids

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
              "FamilySize", "Title", "FamilyId"]
selector = SelectKBest(f_classif, k=5)
selector.fit(df_titanic[predictors], df_titanic["Survived"])
scores = -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=150,
                             min_samples_split=8, min_samples_leaf=4)

scores = cross_validation.cross_val_score(alg, df_titanic[predictors],
                                          df_titanic.Survived, cv=3)

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
     ["Pclass", "Sex", "Age", "Fare", "Embarked",
      "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare",
                                          "FamilySize", "Title", "Age",
                                          "Embarked"]]
]
