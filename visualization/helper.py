import numpy as np

def mean_and_error(MAE_test_end_values,x):
	# Calc mean and errorbars for points of multible runs
	sorting_index = np.argsort(x) # x must be sorted
	x = x[sorting_index]
	MAE_test_end_values = MAE_test_end_values[sorting_index]
	x_new = []
	MAE_new = []
	MAE_err = []
	i = 0
	x = np.append(x,-1)
	while i<len(x)-1:
		print(i)
		for j in range(i,len(x)):
			#print(i,j, x[i], x[j])
			#print(x[i], x[j])
			if x[i] == x[j] and j != len(x)-1:
				continue
			else:
				#print(x[i])
				x_new.append(x[i])
				MAE_new.append(np.mean(MAE_test_end_values[i:j])) # Not including j
				MAE_err.append(np.std(MAE_test_end_values[i:j]))
				i = j
				break

	return x_new,MAE_new,MAE_err

# If last_step=0 take all steps into account
def get_min_MAEs(mls,last_step=0):
	MAE_test_end_values = np.array([])
	bad_index = np.array([], dtype=int)
	for i,ml in enumerate(mls):
		steps_index = int(last_step/ml.params["summary_interval"])
		MAE = np.array(ml.get_AME_test())
		#print(len(MAE)-1)
		#print(steps_index)
		#print()
		if len(MAE)-1 < 5: # if empty
			bad_index = np.append(bad_index, i)
			continue
		if last_step == 0:
			MAE_min_i = np.min(MAE)
		else:
			if len(MAE)-1 < steps_index: # The whole thing
				MAE_min_i = np.min(MAE)
			else:
				MAE_min_i = np.min(MAE[0:steps_index])
		MAE_test_end_values = np.append(MAE_test_end_values, MAE_min_i)

	return MAE_test_end_values,bad_index
