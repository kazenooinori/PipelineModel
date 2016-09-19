from __future__ import print_function
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import numpy as np
import sys

from sklearn import linear_model
from sklearn.svm import SVR

p_types = ['"MACHINE_MOTOR_POWER"', '"MACHINE_MOTOR_RPM"',
		  '"SUPPLY_MOTOR_POWER"', '"SUPPLY_MOTOR_RPM"',
		  '"TAKEUP_MOTOR_RPM_L"', '"TAKEUP_MOTOR_RPM_R"',
		  '"TIC27_RKC_READ_PV"']

def train_lr_model(train_data, train_target, test_data, test_target):

	regr = linear_model.LinearRegression()
	regr.fit(train_data, train_target)

	print("Results of Linear Regression....")
	print("==============================\n")

	# coefficient
	print(regr.coef_)

	# Validation:
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % regr.score(test_data, test_target))

	# Plot outputs
	print(type(test_data), file=sys.stderr)
	plt.scatter(test_data, test_target,  color='black')
	plt.plot(test_data, regr.predict(test_data), color='blue',
			 linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.show()


def train_sknn_regressor(train_data, train_target, test_data, test_target):
	# Now using the sknn regressor
	from sknn.mlp import Regressor, Layer

	nn = Regressor(
		layers=[
			Layer("Rectifier", units=200),
			Layer("Linear")],
		learning_rate=0.02,
		n_iter=10)

	nn.fit(train_data, train_target)
	print("Results of SKNN Regression....")
	print("==============================\n")

	# The coefficients
	print('Coefficients: ', regr.coef_)
	# The mean square error
	print("Residual sum of squares: %.2f"
		  % np.mean((nn.predict(test_data) - test_target) ** 2))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % nn.score(test_data, test_target))

	# Plot outputs
	plt.scatter(test_data, test_target,  color='black')
	plt.plot(test_data, nn.predict(test_data), color='blue',
			 linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.show()


def gen_training_data():
	global p_types
	FEAT_WINDOW_SIZE = 2
	DATE_TIME_FORMAT = '"%Y-%m-%d %H:%M:%S"'

	training_data = []
	training_label = []
	data_window = []
	start_time = ''

	for line in sys.stdin:

		# put data into data window
		parts = line.strip().split(';')

		if len(parts) == (len(p_types) + 2)*2:
			data_time_str = parts[1]
			new_data = { data_time_str: { key:value for key, value in zip(parts[2::2], parts[3::2])} }
			data_window.append(new_data)

			# manage data window
			data_time = datetime.strptime(data_time_str, DATE_TIME_FORMAT)
			window_start_time = data_time - timedelta(seconds=FEAT_WINDOW_SIZE-1)

			new_training_data = [0]*FEAT_WINDOW_SIZE*len(p_types)
			new_head = 0
			for data in data_window:
				window_data_time = datetime.strptime(data.keys()[0], DATE_TIME_FORMAT)
				if window_data_time < window_start_time:
					new_head += 1
				else:
					# gen feature for weight for time window
					time_diff = int((data_time - window_data_time).total_seconds())
					print("data_time:" + str(data_time), file=sys.stderr)
					print("data.keys()[0]:" + str(data.keys()[0]), file=sys.stderr)
					print("time_diff:" + str(time_diff), file=sys.stderr)
					start_index = (FEAT_WINDOW_SIZE - time_diff - 1) * len(p_types)
					print("start index:" + str(start_index), file=sys.stderr)
					for i, t in enumerate(p_types):
						print(str(start_index + i), file=sys.stderr)
						print(str(data.values()[0][t]), file=sys.stderr)
						new_training_data[start_index + i] = data.values()[0][t]

			training_label.append(new_data.values()[0]['"weight"'])
			training_data.append(new_training_data)
			print(new_training_data, file=sys.stderr)

			# clean up obsolete data from head
			if new_head:
				data_window = data_window[new_head:]
		
	return training_label, training_data

def split_data(data, target, train_data_size):
	data = np.array(data).astype(np.float)
	target = np.array(target).astype(np.float)

	total_data_size = len(data) #train_data_size*2
	train_data, test_data = data[:train_data_size], data[train_data_size:]
	train_target, test_target = target[:train_data_size], target[train_data_size:]

	print(train_data)
	print(train_target)
	print("train_data len:" + str(len(train_data)) + "train_target len:" + str(len(train_target)))

	print(test_data)
	print(test_target)
	print("test_data len:" + str(len(test_data)) + "test_target len:" + str(len(test_target)))

	return train_data, train_target, test_data, test_target

def main():
	train_data_size = int(sys.argv[1])

	# prepare data
	target, data = gen_training_data()
	train_data, train_target, test_data, test_target = split_data(data, target, train_data_size)

	# model
	train_lr_model(train_data, train_target, test_data, test_target)
	train_sknn_regressor(train_data, train_target, test_data, test_target)

if __name__	== '__main__':
	main()
