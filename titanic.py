import pandas as pa
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation

## read data
df_passengers = pa.DataFrame(pa.read_csv("train.csv", index_col="PassengerId"))
df_passengers.drop(["Ticket", "Cabin", "Name"], axis=1, inplace=True)

df_passengers.Age.fillna(df_passengers.Age.median(), inplace=True)

df_passengers.loc[df_passengers["Sex"] == "male", "Sex"] = 0
df_passengers.loc[df_passengers["Sex"] == "female", "Sex"] = 1

df_passengers.Embarked.fillna("S", inplace=True)
df_passengers.loc[df_passengers["Embarked"] == "S", "Embarked"] = 0
df_passengers.loc[df_passengers["Embarked"] == "C", "Embarked"] = 1
df_passengers.loc[df_passengers["Embarked"] == "Q", "Embarked"] = 2

## data used for prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

## initalize regression
alg = LinearRegression()

## cross validation folds
kf_passengers = KFold(df_passengers.shape[0], n_folds=4, random_state=1)

predictions = []
for train, test in kf_passengers:
    train_predictors = (df_passengers[predictors].iloc[train, :])
    train_target = df_passengers["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(df_passengers[predictors].iloc[test, :])
    predictions.append(test_predictions)
    
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

accuracy = float((predictions == df_passengers.Survived).sum())/len(predictions)

alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, df_passengers[predictors], df_passengers.Survived, cv=4)

df_test = pa.read_csv("test.csv", index_col="PassengerId")

df_test.Age.fillna(df_passengers.Age.median(), inplace=True)
df_test.loc[df_test.Sex == "male", "Sex"] = 0
df_test.loc[df_test.Sex == "female", "Sex"] = 1
df_test.Embarked.fillna("S", inplace=True)
df_test.loc[df_test.Embarked == "S", "Embarked"] = 0
df_test.loc[df_test.Embarked == "C", "Embarked"] = 1
df_test.loc[df_test.Embarked == "Q", "Embarked"] = 2
df_test.Fare.fillna(df_test.Fare.median(), inplace=True)

alg.fit(df_passengers[predictors], df_passengers.Survived)
predictions = alg.predict(df_test[predictors])

submission = pa.DataFrame({
        "PassengerId": df_test.index.values,
        "Survived": predictions
    })

submission.to_csv("kaggle.csv", index=False)





























