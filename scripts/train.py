from __future__ import print_function
from datetime import datetime, timedelta

import numpy as np
import sys, json
import time, random

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import cross_validation

import logging
logging.basicConfig()

def train_lr_model(data, target):

	print("[Linear Regression]")
	loo = cross_validation.LeaveOneOut(len(target))
	regr = linear_model.LinearRegression()
	scores = cross_validation.cross_val_score(regr, data, target, scoring='neg_mean_squared_error', cv=loo,)

	# This will print the mean of the list of errors that were output and 
	# provide your metric for evaluation
	print("mean of score: %.2f" % scores.mean())
	print("===========================\n")

	print("[2]")
	train_data, test_data = data[0::2], data[1::2]
	train_target, test_target = target[0::2], target[1::2]

	regr = linear_model.LinearRegression()
	regr.fit(train_data, train_target)

	# coefficient
	print(regr.coef_)

	# Validation:
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % regr.score(test_data, test_target))

def train_svr_model(data, target):
	print("[SVR]")
	clf = SVR(C=1.0, epsilon=2.5)
	scores = cross_validation.cross_val_score(
		    clf, data, target, cv=5)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	print("==============================\n")

	"""
	svr_lin = SVR(kernel='linear', epsilon=2.5)
	scores = cross_validation.cross_val_score(
		    svr_lin, data, target, cv=5)
	print("[SVR -linear]")
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	print("==============================\n")

	svr_poly = SVR(kernel='poly', epsilon=2.5, degree=2)
	scores = cross_validation.cross_val_score(
		    svr_poly, data, target, cv=5)
	print("[SVR - poly]")
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	print("==============================\n")
	"""

def train_sknn_regressor(train_data, train_target, test_data, test_target):
	# Now using the sknn regressor
	from sknn.mlp import Regressor, Layer

	nn = Regressor(
		layers=[
			Layer("Rectifier", units=200),
			Layer("Linear")],
		learning_rate=0.00002,
		n_iter=10)

	nn.fit(train_data, train_target)
	print("[SKNN Regression]")

	# The mean square error
	print("Residual sum of squares: %.2f"
		  % np.mean((nn.predict(test_data) - test_target) ** 2))

	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % nn.score(test_data, test_target))

	scores = nn.score(test_data, test_target)
	#scores = cross_validation.cross_val_score(
	#                       nn, data, target, cv=5)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	print("==============================\n")

def main():
	start = time.time()
	# load data
	training_data = json.load(open(sys.argv[1], 'r'))
	if len(sys.argv) > 2 and sys.argv[2] == 'filter':
		training_data = filter(lambda x: x[0] > 7390 or x[0] < 9535, training_data)

	training_data = random.sample(training_data, 50000)
	target, data = zip(*training_data)
	data = np.array(data).astype(np.float)
	target = np.array(target).astype(np.float)

	# model
	print("*** model training ***", file=sys.stderr)
	print("total data size: " + str(len(data)), file=sys.stderr)

	if sys.argv[3] == 'linear':
		train_lr_model(data, target)
	elif sys.argv[3] == 'svr':
		train_svr_model(data, target)
	elif sys.argv[3] == 'nn':
		train_sknn_regressor(train_data, train_target, test_data, test_target)
	else:
		print("No model assigned")

	elasped_time = time.time() - start
	print("elasped time: " + str(elasped_time))

if __name__	== '__main__':
	main()

