#!/usr/bin/env python
"""This script makes prediction on Titanic survivors data."""


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# read data
df_titanic = pd.DataFrame(pd.read_csv("train.csv", index_col="PassengerId"))

# preprocess data
df_titanic.drop(["Ticket", "Cabin", "Name"], axis=1, inplace=True)
df_titanic.Age.fillna(df_titanic.Age.median(), inplace=True)
df_titanic.loc[df_titanic["Sex"] == "male", "Sex"] = 0
df_titanic.loc[df_titanic["Sex"] == "female", "Sex"] = 1
df_titanic.Embarked.fillna("S", inplace=True)
df_titanic.loc[df_titanic["Embarked"] == "S", "Embarked"] = 0
df_titanic.loc[df_titanic["Embarked"] == "C", "Embarked"] = 1
df_titanic.loc[df_titanic["Embarked"] == "Q", "Embarked"] = 2

# data used for prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# normalization of variables used for prediction
for n, index in enumerate(predictors):
    df_titanic[index] = (df_titanic.loc[:, index] - df_titanic.loc[:, index].mean())/df_titanic.loc[:, index].std()

# initialize regression
alg = LogisticRegression(random_state=1)
alg = Pipeline([('poly', PolynomialFeatures(degree=2)),
                ('linear', LogisticRegression(fit_intercept=False, n_jobs=-1))])

scores = cross_validation.cross_val_score(alg,
                                          df_titanic[predictors],
                                          df_titanic.Survived,
                                          cv=3)

print('Accuracy on the training set:', scores.mean())

df_test = pd.read_csv("test.csv", index_col="PassengerId")

df_test.Age.fillna(df_titanic.Age.median(), inplace=True)
df_test.loc[df_test.Sex == "male", "Sex"] = 0
df_test.loc[df_test.Sex == "female", "Sex"] = 1
df_test.Embarked.fillna("S", inplace=True)
df_test.loc[df_test.Embarked == "S", "Embarked"] = 0
df_test.loc[df_test.Embarked == "C", "Embarked"] = 1
df_test.loc[df_test.Embarked == "Q", "Embarked"] = 2
df_test.Fare.fillna(df_test.Fare.median(), inplace=True)
for n, index in enumerate(predictors):
    df_test[index] = (df_test.loc[:, index] - df_test.loc[:, index].mean())/df_test.loc[:, index].std()

alg.fit(df_titanic[predictors], df_titanic.Survived)
predictions = alg.predict(df_test[predictors])

submission = pd.DataFrame({
    "PassengerId": df_test.index.values,
    "Survived": predictions
    })

submission.to_csv("prediction.csv", index=False)
