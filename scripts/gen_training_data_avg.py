from __future__ import print_function
from collections import OrderedDict
from datetime import datetime, timedelta
import numpy as np
import sys, json

p_types = ['MACHINE_MOTOR_POWER', 'MACHINE_MOTOR_RPM',
		  'SUPPLY_MOTOR_POWER', 'SUPPLY_MOTOR_RPM',
		  'TAKEUP_MOTOR_RPM_L', 'TAKEUP_MOTOR_RPM_R',
		  'TIC27_RKC_READ_PV']

p_types = [
		  'TAKEUP_MOTOR_RPM_L', 'TAKEUP_MOTOR_RPM_R',
		  ]


def gen_training_data():
	global p_types
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

				# only get contious data
				data_time = datetime.strptime(data_time_str, DATE_TIME_FORMAT)

				if data_window:
					target_data_time = datetime.strptime(data_window[-1].keys()[0], DATE_TIME_FORMAT) + timedelta(seconds=1)

				if not data_window or data_time == target_data_time:
					data_window.append(new_data)
				else:
					data_window = [new_data]

				if 'W' in new_data.values()[0] and new_data.values()[0]['W']:
					first_data_time = datetime.strptime(data_window[0].keys()[0], DATE_TIME_FORMAT)
					t = int((data_time - first_data_time).total_seconds()) + 1
					w = new_data.values()[0]['W']

					# gen avg for each param
					param_list = [d.values()[0] for d in data_window]
					avg_feat = {}
					for key in new_data.values()[0]:
						if key != 'W' and key in p_types:
							key_arr = [float(d[key]) for d in param_list]
							avg_feat[key] = sum(key_arr)/float(len(key_arr))
					print( 'w: {0}, t: {1}, params:{2}'.format(str(w), str(t), str(avg_feat)), file=sys.stderr)
					
					training_label.append(w)
					training_data.append(OrderedDict(avg_feat).values())

					# reset data_window
					data_window = []

		return training_label, training_data

def main():
	filename = 'training_data/avg_data_w_2f.json'
	f = open(filename, 'w')
	target, data = gen_training_data()
	print(json.dumps(zip(target, data), indent=2), file=f)

if __name__	== '__main__':
	main()

