from visualization.model_loader import *
from feature_data_provider import *
from carbondata import *
import os
import matplotlib.pyplot as plt
import seaborn as sns

fraction_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], dtype=str)
fraction_list_hist = np.array([0.05, 0.2, 0.4, 0.5, 0.7, 1], dtype=str)

forces = {}
forces["multi-perturb"] = CarbonData("/home/bahnsen/carbon_nn/carbondata/MixedCarbon/multi_perturb/", random_seed=None, with_forces=True).data_forces[7999:]
forces["dual-dual"] = CarbonData("/home/bahnsen/carbon_nn/carbondata/MixedCarbon/non_relaxed_double0.8_non_relaxed_double0.2/", random_seed=None, with_forces=True).data_forces[8645:]
forces["dual-relax"] = CarbonData("/home/bahnsen/carbon_nn/carbondata/MixedCarbon/non_relaxed_double0.8_relaxed0.2/", random_seed=None, with_forces=True).data_forces[8645:]
forces["single-single"] = CarbonData("/home/bahnsen/carbon_nn/carbondata/MixedCarbon/non_relaxed_single0.8_non_relaxed_single0.2/", random_seed=None, with_forces=True).data_forces[8645:]
forces["single-relax"] = CarbonData("/home/bahnsen/carbon_nn/carbondata/MixedCarbon/non_relaxed_single0.8_relaxed0.2/", random_seed=None, with_forces=True).data_forces[8645:]

# print("NN forces")
# print(forces_nn[0])
# print("Actual forces")
# print(forces[0])

# delta_forces = forces - forces_nn
# length_delta_forces = np.linalg.norm(delta_forces,axis=2)

# print(delta_forces.shape)
# print(length_delta_forces.shape)

# length_delta_forces = length_delta_forces.flatten()
# print("Mean delta length:", np.mean(length_delta_forces), "eV/Å")

# fig = plt.figure(0)
# plt.hist(length_delta_forces, bins=20)
# plt.xlabel("|DFT force - network force prediction| [eV/Å]")
# plt.ylabel("Population")
# plt.title("Distribution of error in dual-dual point force prediction")
# plt.savefig("dual_dual/dual_dual_hist.pdf")

def length_delta_forces(forces, forces_nn):
	delta_forces = forces - forces_nn
	return np.linalg.norm(delta_forces,axis=2).flatten()

def mean_delta_force(forces, forces_nn):
	return np.mean(length_delta_forces(forces, forces_nn))

# --- Forces error distribution ---
# TODO: layerd histogram, to see it move! (https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0)
# def force_error_destribution(data_name_prefix):
# 	mean_delta_forces = np.ndarray(len(fraction_list))

# 	for i,frac in enumerate(fraction_list):
# 		data_name = data_name_prefix + str(frac) + ".npy"
# 		forces_nn = np.load(data_name)[1:]
# 		lengths = length_delta_forces(forces, forces_nn)
# 		plt.hist(lengths, bins=20)
# 	plt.xlabel("|DFT force - network force prediction| [eV/Å]")
# 	plt.ylabel("Population")
# 	plt.title("Distribution of error in dual-dual point force prediction")

# # Dual-dual
# data_name_prefix = "all_forces_test_dual_dual-29-29-29_struc"
# force_error_destribution(data_name_prefix)
# plt.title("Force learning curve for dual-dual point")
# plt.savefig(data_name_prefix + "_force_learn_curve.pdf")

# # Dual-relax
# data_name_prefix = "all_forces_test_dual_relax-29-29-29_struc"
# force_error_destribution(data_name_prefix)
# plt.title("Force learning curve for dual point-relax")
# plt.savefig(data_name_prefix + "_force_learn_curve.pdf")

# # Single-single
# data_name_prefix = "all_forces_test_single_single-29-29-29_struc"
# force_error_destribution(data_name_prefix)
# plt.title("Force learning curve for single-single point")
# plt.savefig(data_name_prefix + "_force_learn_curve.pdf")

# # Single-relax
# data_name_prefix = "all_forces_test_single_relax-29-29-29_struc"
# force_error_destribution(data_name_prefix)
# plt.title("Force learning curve for single point-relax")
# plt.savefig(data_name_prefix + "_force_learn_curve.pdf")

def force_error_destribution(data_name_prefix,dataset_name):
	forces_real = forces[dataset_name]
	n_dataset = (10000 - len(forces_real))

	plt.clf()
	for i,frac in enumerate(fraction_list_hist):
		data_name = data_name_prefix + frac + ".npy"
		forces_nn = np.load(data_name)
		min_index = min(len(forces_real), len(forces_nn))
		length_delta = length_delta_forces(forces_real[len(forces_real)-min_index+1:], forces_nn[len(forces_nn)-min_index+1:])
		sns.distplot(length_delta, hist = False, kde = True,
                 	 kde_kws = {'shade': False, 'linewidth': 1}, 
                  	 label = frac)

	plt.xlim(-0.5, 7)
	plt.xlabel("|DFT force - network force prediction| [eV/Å]")
	plt.ylabel("Population")

# multi perturb
data_name_prefix = "all_forces_test_multi_perturb-29-29-29_struc"
force_error_destribution(data_name_prefix, "multi-perturb")
plt.title("Force learning curve for multible pertubations")
plt.savefig(data_name_prefix + "_force_hist.pdf")

# --- Learning curve for forces ---
def force_leannig_curve(data_name_prefix,dataset_name):
	forces_real = forces[dataset_name]
	n_dataset = (10000 - len(forces_real))

	mean_delta_forces = np.ndarray(len(fraction_list))

	for i,frac in enumerate(fraction_list):
		data_name = data_name_prefix + frac + ".npy"
		forces_nn = np.load(data_name)
		min_index = min(len(forces_real), len(forces_nn))
		mean_delta = mean_delta_force(forces_real[len(forces_real)-min_index+1:], forces_nn[len(forces_nn)-min_index+1:])
		mean_delta_forces[i] = mean_delta

	plt.clf()
	plt.ylim(1,4)
	plt.plot(n_dataset*fraction_list.astype(float), mean_delta_forces, '.')
	plt.xlabel("Number of training examples")
	plt.ylabel("|DFT force - network force prediction| [eV/Å]")

# multi perturb
data_name_prefix = "all_forces_test_multi_perturb-29-29-29_struc"
force_leannig_curve(data_name_prefix, "multi-perturb")
plt.title("Force learning curve for multible pertubations")
plt.savefig(data_name_prefix + "_force_learn_curve.pdf")

# Dual-dual
data_name_prefix = "all_forces_test_dual_dual-29-29-29_struc"
force_leannig_curve(data_name_prefix, "dual-dual")
plt.title("Force learning curve for dual-dual point")
plt.savefig(data_name_prefix + "_force_learn_curve.pdf")

# # Dual-relax
data_name_prefix = "all_forces_test_dual_relax-29-29-29_struc"
force_leannig_curve(data_name_prefix, "dual-relax")
plt.title("Force learning curve for dual point-relax")
plt.savefig(data_name_prefix + "_force_learn_curve.pdf")

# # Single-single
data_name_prefix = "all_forces_test_single_single-29-29-29_struc"
force_leannig_curve(data_name_prefix, "single-single")
plt.title("Force learning curve for single-single point")
plt.savefig(data_name_prefix + "_force_learn_curve.pdf")

# # Single-relax
data_name_prefix = "all_forces_test_single_relax-29-29-29_struc"
force_leannig_curve(data_name_prefix, "single-relax")
plt.title("Force learning curve for single point-relax")
plt.savefig(data_name_prefix + "_force_learn_curve.pdf")

