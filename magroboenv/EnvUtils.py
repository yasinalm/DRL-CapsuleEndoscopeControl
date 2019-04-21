import numpy as np

def find_currents_distance(curr1, curr2):
	a = np.array(curr1)
	b = np.array(curr2)

	return np.linalg.norm(a-b)

def find_config_distance(config1, config2):
	a = np.array(config1)
	b = np.array(config2)

	return np.linalg.norm(a-b)

def calculate_reward(goal_config, slave_config, last_slave_config):
		num = find_config_distance(last_slave_config, slave_config)
		dnum = find_config_distance(goal_config, slave_config)

		print("Slave Dist: {} ; Last Dist: {}".format(num,dnum))

		percentage_error = 100.0* (num/dnum)
		print("Error={}%".format(percentage_error ))

		assert(percentage_error >= 0, "Percentage error is negative")

		# 10 percent
		# if percentage_error <= 10:
		# 	 rew = (100.0 - percentage_error*10.0)
		# else:
		# 	 rew = (percentage_error-10.0)*(-1)

		# 50 percent
		# if percentage_error <= 50:
		# 	 rew = 100.0 - (percentage_error*2.0)
		# else:
		# 	 rew = (percentage_error-50.0)*(-1)

		# Absolute
		# rew = (100.0 - percentage_error)

		# Angular
		if dnum <= 100:
			 rew = ((50.0 - dnum)/50.0)
		else:
			#  rew = (50.0-dnum)/50.0
			rew = -1

		return dnum, rew