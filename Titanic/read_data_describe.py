# -*- coding: utf-8 -*-

import pandas as pd

titanic = pd.read_csv("train.csv")
titanic["Age"].fillna(titanic["Age"].median())


print(titanic["Sex"].unique())

titanic.loc[titanic["Sex"] == "male", "Sex"] = 1
titanic.loc[titanic["Sex"] == "female", "Sex"] = 0

print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
