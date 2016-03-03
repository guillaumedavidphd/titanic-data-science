#!/usr/bin/env python
"""This script makes prediction on Titanic survivors data."""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot')
plt.rcParams['figure.figsize'] = (18, 8)
mpl.rcParams['figure.dpi'] = 300

# "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale RGB values to [0, 1] range.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

# read data
df_titanic = sns.load_dataset('titanic')

# some insight on data
grp = df_titanic.groupby('sex')[['survived']].mean()
grp.plot.bar(color=tableau20[0])
grp = df_titanic.pivot_table('survived', index='sex', columns='class')
grp.plot.bar(color=tableau20)
age = pd.cut(df_titanic['age'], [0, 18, 80])
df_titanic.pivot_table('survived', ['sex', age], 'class').unstack().plot.bar(color=tableau20)
fare = pd.qcut(df_titanic['fare'], 2)
df_titanic.pivot_table('survived', ['sex', age], [fare, 'class'])
df_titanic.pivot_table(index='sex', columns='class',
                       aggfunc={'survived': sum, 'fare': 'mean'})
df_titanic.pivot_table('survived', index='sex', columns='class', margins=True).plot.bar(color=tableau20)

# preprocess data
# df_titanic.drop(["Ticket", "Cabin", "Name"], axis=1, inplace=True)
df_titanic.age.fillna(df_titanic.age.median(), inplace=True)
df_titanic.loc[df_titanic["sex"] == "male", "sex"] = 0
df_titanic.loc[df_titanic["sex"] == "female", "sex"] = 1
df_titanic.embarked.fillna("S", inplace=True)
df_titanic.loc[df_titanic["embarked"] == "S", "embarked"] = 0
df_titanic.loc[df_titanic["embarked"] == "C", "embarked"] = 1
df_titanic.loc[df_titanic["embarked"] == "Q", "embarked"] = 2

# data used for prediction
predictors = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

# normalization of variables used for prediction
for n, index in enumerate(predictors):
    df_titanic[index] = (df_titanic.loc[:, index] - df_titanic.loc[:, index].mean())/df_titanic.loc[:, index].std()

# initialize regression
alg = LinearRegression()

# cross validation folds
kf_passengers = KFold(df_titanic.shape[0], n_folds=4, random_state=1)

predictions = []
for train, test in kf_passengers:
    train_predictors = (df_titanic[predictors].iloc[train, :])
    train_target = df_titanic["survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(df_titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

accuracy = float((predictions == df_titanic.survived).sum())/len(predictions)
print('Accuracy on the training set:', accuracy)

alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg,
                                          df_titanic[predictors],
                                          df_titanic.survived,
                                          cv=3)

df_test = pd.read_csv("test.csv", index_col="PassengerId")

df_test.columns = [word.lower() for word in df_test.columns]
df_test.age.fillna(df_titanic.age.median(), inplace=True)
df_test.loc[df_test.sex == "male", "sex"] = 0
df_test.loc[df_test.sex == "female", "sex"] = 1
df_test.embarked.fillna("S", inplace=True)
df_test.loc[df_test.embarked == "S", "embarked"] = 0
df_test.loc[df_test.embarked == "C", "embarked"] = 1
df_test.loc[df_test.embarked == "Q", "embarked"] = 2
df_test.fare.fillna(df_test.fare.median(), inplace=True)
for n, index in enumerate(predictors):
    df_test[index] = (df_test.loc[:, index] - df_test.loc[:, index].mean())/df_test.loc[:, index].std()

alg.fit(df_titanic[predictors], df_titanic.survived)
predictions = alg.predict(df_test[predictors])

submission = pd.DataFrame({
    "PassengerId": df_test.index.values,
    "Survived": predictions
    })

submission.to_csv("kaggle.csv", index=False)
