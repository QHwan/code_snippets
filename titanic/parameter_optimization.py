import pandas as pd
import numpy as np
import re
import os
import sklearn
import pickle

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

train = pd.read_csv(os.path.join('data','engineered_train.csv'))
test = pd.read_csv(os.path.join('data','engineered_test.csv'))
PassengerId = pd.read_csv(os.path.join('data','passengerid.csv'))


ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, random_state=SEED)

y_train = train["Survived"].ravel()
train = train.drop(["Survived"], axis=1)
x_train = train.values
x_test = test.values

class GridSearchHelper(object):
	def __init__(self, clf, seed=0, param_grid=None):
		#params["random_state"] = seed
		seed = 7
		nfolds = 5
		self.kf = KFold(n_splits=nfolds, random_state=seed)
		self.clf = clf()
		self.param_grid = param_grid

	def train(self, x_train, y_train):
		self.clf.fit(x_train, y_train)

	def predict(self, x):
		return self.clf.predict(x)

	def fit(self, x, y):
		return self.clf.fit(x,y)

	def feature_importances(self,x,y):
		print(self.clf.fit(x,y).feature_importances_)

	def grid_search(self, x, y):
		return GridSearchCV(self.clf, param_grid=self.param_grid, cv=self.kf, scoring="accuracy", n_jobs=4, verbose=0).fit(x,y).best_params_



rf_params = {
	"n_estimators": [500],
	"warm_start": [True],
	"max_depth": [6],
	"min_samples_leaf": [2],
	"max_features": ["sqrt"],
}

et_params = {
	"n_jobs": [-1],
	"n_estimators": [500],
	"max_depth": [8],
	"min_samples_leaf": [2],
	"verbose": [0]
}

ada_params = {
	"n_estimators": [500],
	"learning_rate": [0.75]
}

gb_params = {
	"n_estimators": [500],
	"max_depth": [5],
	"min_samples_leaf": [2],
	"verbose": [0]
}

svc_params = {
	"kernel": ["linear"],
	"C": [0.025]
}

log_params = {
	"solver": ["liblinear"]
}

kn_params = {
	"algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
}

mlp_params = {
	"alpha": [1]
}



CLASSIFIERS = [ [AdaBoostClassifier, ada_params, 'ada'],
		[ExtraTreesClassifier, et_params, 'et'],
		[GradientBoostingClassifier, gb_params, 'gb'],
		[KNeighborsClassifier, kn_params, 'kn'],
		[LogisticRegression, log_params, 'log'],
		[MLPClassifier, mlp_params, 'mlp'],
		[RandomForestClassifier, rf_params, 'rf'],
		[SVC, svc_params, 'svc']]


for classifier in CLASSIFIERS:
	clf, params, name = classifier

	grid_helper = GridSearchHelper(clf=clf, seed=SEED, param_grid=params)
	best_params = grid_helper.grid_search(x_train, y_train)

	print(best_params)

	params_filename = os.path.join('./best_parameters','best_'+name+'_params.p')
	with open(params_filename, "wb") as f:
		pickle.dump(best_params, f)
	f.close()

	 