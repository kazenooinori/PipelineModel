from __future__ import print_function
from datetime import datetime, timedelta
import sys, json

p_types = ['MACHINE_MOTOR_POWER', 'MACHINE_MOTOR_RPM',
		  'SUPPLY_MOTOR_POWER', 'SUPPLY_MOTOR_RPM',
		  'TAKEUP_MOTOR_RPM_L', 'TAKEUP_MOTOR_RPM_R',
		  'TIC27_RKC_READ_PV']

def gen_training_data(feat_window_size):
	global p_types
	FEAT_WINDOW_SIZE = feat_window_size 
	DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

	training_data = []
	training_label = []
	data_window = []
	start_time = ''

	with open(sys.argv[1]) as f:
		for line in f.readlines():

			# put data into data window
			parts = line.strip().replace('"', '').replace(";;", ";0;").split(';')

			if len(parts) > (len(p_types))*2 + 1:
				data_time_str = parts[0]
				if not parts[-1]:
					parts = parts[:-1]
				new_data = { data_time_str: { key:value for key, value in zip(parts[1::2], parts[2::2])} }
				data_window.append(new_data)

				if 'W' in new_data.values()[0] and new_data.values()[0]['W']:
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
							start_index = (FEAT_WINDOW_SIZE - time_diff - 1) * len(p_types)
							for i, t in enumerate(p_types):
								new_training_data[start_index + i] = data.values()[0][t]

					training_label.append(new_data.values()[0]['W'])
					training_data.append(new_training_data)

					# clean up obsolete data from head
					if new_head:
						data_window = data_window[new_head:]

		return training_label, training_data

def main(feat_window_size):
	filename = 'training_data/data_{0}.json'.format(str(feat_window_size))
	f = open(filename, 'w')
	target, data = gen_training_data(feat_window_size)
	print(json.dumps(zip(target, data), indent=2), file=f)

if __name__	== '__main__':
	for _ in xrange(2, 41):
		main(_)
