import glob
import shelve
import numpy as np
import matplotlib.pyplot as plt

def sparsity(vals):
	assert np.abs(sum(vals) - 100) < 1, vals
	vals = [x / 100 for x in vals]
	return ((sum(vals)/len(vals)) ** 2) / (sum([x**2 for x in vals]) / len(vals))

neurons_with_shap = glob.glob("cache/*/shap*")
neurons = []
for n in neurons_with_shap:
	nid = n.split('\\')[1]
	try:
		neurons.append(int(nid))
	except:
		pass
neurons = sorted(list(set(neurons)))
print("number_of_cells:", len(neurons))

def get_sparsities(allo_neurons, ego_neurons):
	sparsity_index = []
	flag = 0
	for t, neurons in enumerate([allo_neurons, ego_neurons]):
		if t == 0:
			n_features = 6
		if t == 1:
			n_features = 8

		mat = np.zeros((len(neurons), n_features))
		keys = []
		for i, n in enumerate(neurons):
			cache = shelve.open(f"cache/{n}/shap")
			# print(list(cache.keys()))
			#if len(list(cache.keys())) == 0: continue
			#if len(list(cache.keys())) == 1 and n_features == 8: continue
			if len(list(cache.keys())) == 0: continue
			model_key = list(cache.keys())[0]
			shapley = cache[model_key][0]
			shapley_s = dict(sorted(shapley.items(), key=lambda item: item[flag], reverse=bool(flag)))
			keys = list(shapley_s.keys())
			keys = [k.replace("_F_", "_") for k in keys]
			shapley_s = list(shapley_s.values())
			if not flag and n_features == 6:
				#print(keys)
				keys = [keys[1], keys[0]] + keys[2:]
				#print(keys)
				shapley_s = [shapley_s[1], shapley_s[0]] + shapley_s[2:]
			mat[i] = shapley_s
			# print(shapley_s, n)
			val = sparsity(shapley_s)
			sparsity_index.append(val)
			#print(sparsity_index)
	perm = sorted(zip(sparsity_index, range(len(sparsity_index))))
	sorted_sparsity, perm = list(zip(*perm))
	# print(sorted_sparsity)
	allo_sparsities = []
	ego_sparsities = []
	for i in perm:
		if i < len(allo_neurons):
			allo_sparsities.append(sorted_sparsity.index(sparsity_index[i])+1)
		else:
			ego_sparsities.append(sorted_sparsity.index(sparsity_index[i])+1)
	return allo_sparsities, ego_sparsities

def plot_analysis(neurons, n_features, flag, neuron_ids):
	mat = np.zeros((len(neurons), n_features))
	sparsity_index = []
	keys = []
	for i, n in enumerate(neurons):
		cache = shelve.open(f"cache/{n}/shap")
		print(list(cache.keys()))
		if len(list(cache.keys())) == 0: continue
		if len(list(cache.keys())) == 1 and n_features == 8: continue
		model_key = list(cache.keys())[n_features == 8]
		shapley = cache[model_key][0]
		shapley_s = dict(sorted(shapley.items(), key=lambda item: item[flag], reverse=bool(flag)))
		keys = list(shapley_s.keys())
		keys = [k.replace("_F_", "_") for k in keys]
		shapley_s = list(shapley_s.values())
		if not flag and n_features == 6:
			#print(keys)
			keys = [keys[1], keys[0]] + keys[2:]
			#print(keys)
			shapley_s = [shapley_s[1], shapley_s[0]] + shapley_s[2:]
		mat[i] = shapley_s
		# print(shapley_s, n)
		val = sparsity(shapley_s)
		sparsity_index.append(val)
		#print(sparsity_index)

	fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))

	# sort matrix
	perm = sorted(zip(sparsity_index, range(len(sparsity_index))))
	#print(len(perm))
	sorted_sparsity, perm = list(zip(*perm))
	#print(perm)
	im = ax.matshow(mat[list(perm)], vmin=0)
	# im2 = ax2.matshow(mat[list(perm)])
	sim_files = ['place_distance4_cell.csv', 'ego_cell4.csv', 'A_place_cell.csv', 'place_distance2_cell.csv', 'ego_cell_1_2.csv', 'ego_cell2.csv', 'place_distance1_cell.csv', 'place_distance3_cell.csv', 'pairwise_distance2,3_cell.csv', 'pairwise_distance1,4_cell.csv', 'randomly_firing_cell.csv', 'pairwise_distance1,3_cell.csv', 'ego_cell1.csv', 'ego_cell3.csv', 'HD_place_cell.csv', 'pairwise_distance1,2_cell.csv', 'pairwise_distance2,4_cell.csv', 'HD_place_cell_pos1.csv', 'pairwise_distance3,4_cell.csv']
	neuron_names = dict(zip(range(len(sim_files)), sim_files))
	neuron_ids2 = list(np.array(neurons)[list(perm)])
	if np.mean(neuron_ids) > 1000:
		ax.set_yticklabels([neuron_names[x%1000] for x in neuron_ids])
	else:
		#ax.set_yticklabels(range(1, len(neuron_ids)+1))
		ax.set_yticklabels(neuron_ids2)
	#ax.set_yticklabels(map(lambda x: f"{x:.3}", sorted_sparsity))
	ax.set_yticks(range(len(sparsity_index)))
	if flag == 0:
		ax.set_xticks(range(len(keys)))
		ax.set_xticklabels(keys, rotation=70, fontsize=6)
	if flag == 1:
		ax.set_xticks(range(len(keys)))
		ax.set_xticklabels(range(1, len(keys)+1))
	if flag == 1:
		ax.set_xlabel("Ranked Variables")
	else:
		ax.set_xlabel("Variables")
	ax.set_ylabel("Neuron id")
	# ax2.set_yticklabels(map(lambda x: f"{x:.3}", sorted_sparsity))
	#ax2.set_yticks(range(len(sparsity_index)))
	ax.autoscale(False)
	#ax2.autoscale(False)
	#print(keys)
	cbar = fig.colorbar(im)
	cbar.ax.set_title("Variable Importance\n(Shapley Values)", fontsize=5)


	ax.set_aspect('auto')

	ax2.plot(sorted(sparsity_index), range(1,1+len(sorted(neuron_ids))))
	ax2.invert_yaxis()
	ax2.set_yticks(range(1,1+len(sorted(neuron_ids))))
	ax2.set_yticklabels(sorted(neuron_ids))

	rs = []
	for i, n in enumerate(neurons):
		cache2 = shelve.open(f"cache/{n}/models")
		if n_features == 6:
			model_key = f"AlloModel|['BAT_0_F_HD', 'BAT_0_F_X', 'BAT_0_F_Y', 'BAT_1_F_X', 'BAT_1_F_Y', 'BAT_2_F_X', 'BAT_2_F_Y', 'BAT_3_F_X', 'BAT_3_F_Y', 'BAT_4_F_X', 'BAT_4_F_Y']|{n}|0"
		elif n_features == 8:
			model_key = f"EgoModel|['BAT_1_F_A', 'BAT_1_F_D', 'BAT_2_F_A', 'BAT_2_F_D', 'BAT_3_F_A', 'BAT_3_F_D', 'BAT_4_F_A', 'BAT_4_F_D']|{n}|0"


		import model_maps
		from scipy.stats import pearsonr
		import data_manager
		from conf import Conf
		import rate_maps
		import pandas as pd

		data = data_manager.Loader7()(n)
		dataprop = data_manager.DataProp1(data)
		m = cache2[model_key]
		data_maps = rate_maps.build_maps(dataprop)

		test_idx = m.model.X_test.index
		model_fr_map = model_maps.ModelFiringRate(dataprop, m.model, m.shuffle_index, test_idx)
		my_model_maps = model_maps.build_maps(m.model, data_maps)

		filter_width = Conf().TIME_BASED_GROUP_SPLIT
		spikes_count = dataprop.orig_spikes_count
		smooth_fr = np.convolve(spikes_count, [1] * filter_width, mode='same') / filter_width
		smooth_fr_no_nans = pd.Series(smooth_fr).loc[dataprop.no_nans_indices]
		smooth_fr_no_nans_shuffled = np.roll(smooth_fr_no_nans, -m.shuffle_index)
		smooth_fr_no_nans_shuffled = smooth_fr_no_nans_shuffled[test_idx]
		shuffle_indices = (dataprop.no_nans_indices[test_idx] + m.shuffle_index) % np.max(dataprop.no_nans_indices[test_idx])
		fr_map = rate_maps.FiringRate(smooth_fr_no_nans_shuffled, shuffle_indices)
		fr_map.process()
		pearson_correlation = pearsonr(fr_map.y, model_fr_map.y)[0]

		rs.append(pearson_correlation)

	ax3.plot(np.array(rs)[list(perm)], range(1,1+len(sorted(neuron_ids))))
	ax3.invert_yaxis()
	ax3.set_yticks(range(1,1+len(sorted(neuron_ids))))
	ax3.set_yticklabels(sorted(neuron_ids))

	ax2.set_xlim((0.6, 1))
	ax3.set_xlim((0.2, 0.8))

	output_name = 'population' + str(int(flag)) + ["Allo", "Ego"][n_features == 8]
	plt.savefig(f'C:/tmp/cosyne-figs/{output_name}.svg', bbox_inches='tight', dpi=300) 
	plt.savefig(f'C:/tmp/cosyne-figs/{output_name}.pdf', bbox_inches='tight', dpi=300)   
	# plt.show()

allo_neurons = []
ego_neurons = []

for i, n in enumerate(neurons):
	ego_neurons.append(n)
	allo_neurons.append(n)
	continue
	if n >= 1000: continue
	cache = shelve.open(f"cache/{n}/shap")
	cache2 = shelve.open(f"cache/{n}/models")

	best_model = None
	best_dic = 10 ** 8
	model_key1 = f"AlloModel|['BAT_0_F_HD', 'BAT_0_F_X', 'BAT_0_F_Y', 'BAT_1_F_X', 'BAT_1_F_Y', 'BAT_2_F_X', 'BAT_2_F_Y', 'BAT_3_F_X', 'BAT_3_F_Y', 'BAT_4_F_X', 'BAT_4_F_Y']|{n}|0"
	model_key2 = f"EgoModel|['BAT_1_F_A', 'BAT_1_F_D', 'BAT_2_F_A', 'BAT_2_F_D', 'BAT_3_F_A', 'BAT_3_F_D', 'BAT_4_F_A', 'BAT_4_F_D']|{n}|0"

	bias = 0
	try:
		m1_dic = cache2[model_key1].model.score[1] - bias
		m2_dic = cache2[model_key2].model.score[1]
	except:
		
		continue
	m1_aic = cache2[model_key1].model.gam_model.statistics_['AICc'] - bias
	m2_aic = cache2[model_key2].model.gam_model.statistics_['AICc']
	delta_dic = m1_dic - m2_dic
	delta_aic = m1_aic - m2_aic
	# print(n, delta_dic, delta_aic)
	if abs(delta_dic) > 10 and (delta_aic * delta_dic > 0):
		best_model = model_key1
		if delta_dic > 0:
			best_model = model_key2
	else:
		continue

	for model_key in [best_model]:
		if model_key.startswith('Ego'):
			ego_neurons.append(n)
		elif model_key.startswith('Allo'):
			allo_neurons.append(n)
		else:
			# print("BUG")
			exit()

print("Allo", allo_neurons)
print("Ego", ego_neurons)
#ego_neurons += ego_neurons[-3:]
allo_sparsities, ego_sparsities = get_sparsities(allo_neurons, ego_neurons)
#plt.plot(allo_sparsities[::-1])
#plt.show()
#plt.plot(ego_sparsities[::-1])
#plt.show()
plot_analysis(allo_neurons, 6, 1, allo_sparsities)
plot_analysis(allo_neurons, 6, 0, allo_sparsities)
plot_analysis(ego_neurons, 8, 1, ego_sparsities)
plot_analysis(ego_neurons, 8, 0, ego_sparsities)