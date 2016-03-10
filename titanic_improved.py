#!/usr/bin/env python
"""This script makes prediction on Titanic survivors data."""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
import re
import operator
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#%% Functions
def get_title(name):
    """Get title of person (Mr., Ms., etc)."""
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

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

#%% Random forest prediction
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

#%% Adding features
df_titanic["FamilySize"] = df_titanic.SibSp + df_titanic.Parch
df_titanic["NameLength"] = df_titanic.Name.apply(lambda x: len(x))

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

family_ids = df_titanic.apply(get_family_id, axis=1)

family_ids[df_titanic["FamilySize"] < 3] = -1

df_titanic["FamilyId"] = family_ids

#%% Feature selection
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
              "FamilySize", "Title", "FamilyId"]
selector = SelectKBest(f_classif, k=5)
selector.fit(df_titanic[predictors], df_titanic["Survived"])
scores = -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.title("Feature selection: select 4 features with highest score")
plt.show()

#%% Predictions
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=150,
                             min_samples_split=8, min_samples_leaf=4)

scores = cross_validation.cross_val_score(alg, df_titanic[predictors],
                                          df_titanic.Survived, cv=3)
print(scores.mean())

#%% Gradient boosting
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
     ["Pclass", "Sex", "Age", "Fare", "Embarked",
      "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare",
                                          "FamilySize", "Title", "Age",
                                          "Embarked"]]
]

kf = KFold(df_titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = df_titanic["Survived"].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(df_titanic[predictors].iloc[train,:], train_target)
        test_predictions = alg.predict_proba(df_titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

accuracy = sum(predictions[predictions == df_titanic["Survived"]]) / len(predictions)
print(accuracy)

#%% Preprocess test set
df_test = pd.read_csv("test.csv", index_col="PassengerId")
df_test.Age.fillna(df_titanic.Age.median(), inplace=True)
df_test.loc[df_test.Sex == "male", "Sex"] = 0
df_test.loc[df_test.Sex == "female", "Sex"] = 1
df_test.Embarked.fillna("S", inplace=True)
df_test.loc[df_test.Embarked == "S", "Embarked"] = 0
df_test.loc[df_test.Embarked == "C", "Embarked"] = 1
df_test.loc[df_test.Embarked == "Q", "Embarked"] = 2
df_test.Fare.fillna(df_test.Fare.median(), inplace=True)

titles = df_test["Name"].apply(get_title)
# We're adding the Dona title to the mapping, because it's in the test set, but not the training set
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
df_test["Title"] = titles

df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"]

family_ids = df_test.apply(get_family_id, axis=1)
family_ids[df_test["FamilySize"] < 3] = -1
df_test["FamilyId"] = family_ids
df_test["NameLength"] = df_test["Name"].apply(lambda x: len(x))

#%% Predictions
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    alg.fit(df_titanic[predictors], df_titanic["Survived"])
    predictions = alg.predict_proba(df_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions<=.5] = 0
predictions[predictions>.5] = 1
predictions = predictions.astype(int)

submission = pd.DataFrame({
    "PassengerId": df_test.index.values,
    "Survived": predictions
    })

submission.to_csv("kaggle.csv", index=False)







































