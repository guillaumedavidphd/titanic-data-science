#!/usr/bin/env python
"""This script makes prediction on Titanic survivors data."""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 8)
mpl.rcParams['figure.dpi'] = 300

#%% "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

#%% Scale RGB values to [0, 1] range.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

#%% read data
df_titanic = pd.DataFrame(pd.read_csv("train.csv", index_col="PassengerId"))

#%% preprocess data
df_titanic.drop(["Ticket", "Cabin", "Name"], axis=1, inplace=True)
df_titanic.Age.fillna(df_titanic.Age.median(), inplace=True)
df_titanic.loc[df_titanic["Sex"] == "male", "Sex"] = 0
df_titanic.loc[df_titanic["Sex"] == "female", "Sex"] = 1
df_titanic.Embarked.fillna("S", inplace=True)
df_titanic.loc[df_titanic["Embarked"] == "S", "Embarked"] = 0
df_titanic.loc[df_titanic["Embarked"] == "C", "Embarked"] = 1
df_titanic.loc[df_titanic["Embarked"] == "Q", "Embarked"] = 2

#%% data used for prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#%% normalization of variables used for prediction
for n, index in enumerate(predictors):
    df_titanic[index] = (df_titanic.loc[:, index] - df_titanic.loc[:, index].mean())/df_titanic.loc[:, index].std()

#%% cross validation folds
kf_passengers = KFold(df_titanic.shape[0], n_folds=5, shuffle=True, random_state=1)

#%% cross validation scores vs. degree polynomial regression
score_vc = []
degrees = np.arange(1, 9)
for n in degrees:
    score = 0
    print(n)
    predictions = []
    alg = Pipeline([('poly', PolynomialFeatures(degree=n)),
                    ('linear', LinearRegression(fit_intercept=False, n_jobs=-1))])
    for train, test in kf_passengers:
        train_predictors = (df_titanic[predictors].iloc[train, :])
        train_target = df_titanic["Survived"].iloc[train]
        test_target = df_titanic["Survived"].iloc[test]
        alg.fit(train_predictors, train_target)
        # prediction on test set to calculate test MSE
        test_predictions = alg.predict(df_titanic[predictors].iloc[test, :])
        test_predictions[test_predictions > .5] = 1
        test_predictions[test_predictions <= .5] = 0
        predictions.append(test_predictions)
        score += np.sum((test_target - test_predictions)**2)
    score_vc.append(np.sqrt(1/len(df_titanic["Survived"])*score))
#        predictions.append(test_predictions)

plt.figure()
ax = plt.subplot(111)
scorevc = plt.plot(degrees, score_vc, 'o', color=tableau20[6])
plt.xticks([])
plt.yticks(fontsize=14)
plt.xlabel('Degree', fontsize=16, fontweight='bold')
plt.ylabel('Cross-validation score', fontsize=16, fontweight='bold')
plt.title('Polynomial regressions\n', fontsize=22, fontweight='bold')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_axis_bgcolor('white')
for degree in degrees:
    plt.text(degree, score_vc[degree-1]*.975, str(degree), ha='center')
plt.xlim(.5, 8.5)
plt.ylim(.39, .66)
ax.yaxis.grid(color='k', alpha=.3, linestyle='dashed')
plt.savefig("CVscoreVSdegree.png",
            bbox_inches='tight',
            dpi=300,
            format='png')
plt.savefig("CVscoreVSdegree.pdf",
            bbox_inches='tight',
            dpi=300,
            format='pdf')
