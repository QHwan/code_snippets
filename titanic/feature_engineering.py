import pandas as pd
import numpy as np
import re
import os
import sklearn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Going to use these 5 base models for the stacking
import xgboost as xgb

from sklearn.ensemble import (RandomForestClassifier, 
							AdaBoostClassifier,
							GradientBoostingClassifier,
							ExtraTreesClassifier,
							VotingClassifier)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score


train = pd.read_csv(os.path.join('data','train.csv'))
test = pd.read_csv(os.path.join('data','test.csv'))

PassengerId = test['PassengerId']


full_data = [train, test]

train["Name_length"] = train["Name"].apply(len)
test["Name_length"] = test["Name"].apply(len)

train["Has_Cabin"] = train["Cabin"].apply(lambda x: 0 if type(x)==float else 1)
test["Has_Cabin"] = test["Cabin"].apply(lambda x: 0 if type(x)==float else 1)

for dataset in full_data:
	dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

	dataset["IsAlone"] = 0
	dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1

	dataset["Embarked"] = dataset["Embarked"].fillna("S")

	dataset["Fare"] = dataset["Fare"].fillna(train["Fare"].median())
train["CategoricalFare"] = pd.qcut(train["Fare"], 4)

for dataset in full_data:
	age_avg = dataset["Age"].mean()
	age_std = dataset["Age"].std()
	age_null_count = dataset["Age"].isnull().sum()
	age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
	dataset["Age"][np.isnan(dataset["Age"])] = age_null_random_list
	dataset["Age"] = dataset["Age"].astype(int)
train["CategoricalAge"] = pd.cut(train["Age"], 5)

def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
	dataset["Title"] = dataset["Name"].apply(get_title)
	dataset["Title"] = dataset["Title"].replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
	dataset["Title"] = dataset["Title"].replace("Mlle", "Miss")
	dataset["Title"] = dataset["Title"].replace("Ms", "Miss")
	dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")

	dataset["Sex"] = dataset["Sex"].map({"female":0, "male":1}).astype(int)

	title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
	dataset["Title"] = dataset["Title"].map(title_mapping)
	dataset["Title"] = dataset["Title"].fillna(0)

	dataset["Embarked"] = dataset["Embarked"].map({"S":0, "C":1, "Q":2}).astype(int)

	dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0
	dataset.loc[(dataset["Fare"]>7.91) & (dataset["Fare"]<=14.454), "Fare"] = 1
	dataset.loc[(dataset["Fare"]>14.454) & (dataset["Fare"]<=31), "Fare"] = 2
	dataset.loc[(dataset["Fare"]>31), "Fare"] = 3
	dataset["Fare"] = dataset["Fare"].astype(int)

	dataset.loc[dataset["Age"] <= 16, "Age"] = 0
	dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
	dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
	dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
	dataset.loc[(dataset["Age"] > 64), "Age"] = 1

drop_elements = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp"]
train = train.drop(drop_elements, axis=1)
train = train.drop(["CategoricalAge", "CategoricalFare"], axis=1)
test = test.drop(drop_elements, axis=1)

train.to_csv(os.path.join("data","engineered_train.csv"), index=False)
test.to_csv(os.path.join("data","engineered_test.csv"), index=False)
PassengerId.to_csv(os.path.join("data","passengerid.csv"), index=False, header=True)



