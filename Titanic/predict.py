# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold


titanic = pd.read_csv("train.csv")

print(titanic.head(5))

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

print(titanic["Sex"].unique())

titanic.loc[titanic["Sex"] == "male", "Sex"] = 1
titanic.loc[titanic["Sex"] == "female", "Sex"] = 0

print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0 
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

predictors = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
]

alg = LinearRegression()

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)


predictions = []
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train,:])
    train_target = titanic["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

predictions[predictions == titanic["Survived"]]

