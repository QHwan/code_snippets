import pandas as pd
import numpy as np
import re
import os
import pickle
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

class SKlearnHelper(object):
	def __init__(self, clf, seed=0, params=None):
		#params["random_state"] = seed
		seed=7
		self.clf = clf(**params)

	def train(self, x_train, y_train):
		self.clf.fit(x_train, y_train)

	def predict(self, x):
		return self.clf.predict(x)

	def fit(self, x, y):
		return self.clf.fit(x,y)

	def feature_importances(self,x,y):
		print(self.clf.fit(x,y).feature_importances_)

def get_oof(clf, x_train, y_train, x_test):
	oof_train = np.zeros((ntrain,))
	oof_test = np.zeros((ntest,))
	oof_test_skf = np.empty((NFOLDS, ntest))

	for i, (train_index, test_index) in enumerate(kf.split(x_train)):
		x_tr = x_train[train_index]
		y_tr = y_train[train_index]
		x_te = x_train[test_index]

		clf.train(x_tr, y_tr)

		oof_train[test_index] = clf.predict(x_te)
		oof_test_skf[i, :] = clf.predict(x_test)

	oof_test[:] = oof_test_skf.mean(axis=0)
	return oof_train.reshape(-1,1), oof_test.reshape(-1,1)



CLASSIFIERS = [ [AdaBoostClassifier, 'ada'],
		[ExtraTreesClassifier, 'et'],
		[GradientBoostingClassifier, 'gb'],
		[KNeighborsClassifier, 'kn'],
		[LogisticRegression, 'log'],
		[MLPClassifier, 'mlp'],
		[RandomForestClassifier, 'rf'],
		[SVC, 'svc']]

x_train_stack = []
x_test_stack = []
for classifier in CLASSIFIERS:
	clf, name = classifier
	params_filename = os.path.join('./best_parameters','best_'+name+'_params.p')

	params = {}
	with open(params_filename, "rb") as f:
		params = pickle.load(f)
	f.close()

	print(params)


	instance = SKlearnHelper(clf=clf, seed=SEED, params=params) 

	oof_train, oof_test = get_oof(instance, x_train, y_train, x_test)

	if (x_train_stack == []):
		x_train_stack = np.copy(oof_train)
	if (x_test_stack == []):
		x_test_stack = np.copy(oof_test)

	x_train_stack = np.concatenate((x_train_stack, oof_train), axis=1)
	x_test_stack = np.concatenate((x_test_stack, oof_test), axis=1)




gbm = xgb.XGBClassifier(
	n_estimators=2000,
	max_depth=4,
	min_child_weight=2,
	gamma=0.9,
	subsample=0.8,
	colsample_bytree=0.8,
	objective="binary:logistic",
	nthread=-1,
	scale_pos_weight=1
	).fit(x_train_stack, y_train)
predictions = gbm.predict(x_test_stack)

print(accuracy_score(y_train, gbm.predict(x_train_stack)))

predictions = pd.DataFrame(data=predictions, columns=["Survived"])

submission = pd.concat([PassengerId, predictions], axis=1)
submission.to_csv(os.path.join("data", "submission.csv"), index=False)
