import shelve
import pprint
import numpy as np
import matplotlib.pyplot as plt

nids = [20, 22, 57, 58, 61, 64, 72, 73, 85, 87, 93, 144, 153, 170, 171, 202, 205, 215, 390, 420]
nids1 = [16, 20, 22, 50, 52, 57, 58, 61, 64, 68, 72, 73, 78, 82, 85, 87, 93, 94, 95, 97, 98, 99, 133, 144, 145, 148, 149, 150, 151, 153, 159, 170, 171, 200, 201, 202, 204, 205, 210, 215, 390, 419, 420] 
nids2 = [3, 15, 20, 22, 31, 57, 58, 61, 64, 72, 73, 83, 85, 87, 93, 144, 147, 153, 170, 171, 202, 205, 215, 380, 390, 418, 420, 428]

#nids = [72]
nids = sorted((set(nids1) | set(nids2)) - set([93, 419]))


def sparsity(vals):
	assert np.abs(sum(vals) - 100) < 1, vals
	vals = [x / 100 for x in vals]
	return ((sum(vals)/len(vals)) ** 2) / (sum([x**2 for x in vals]) / len(vals))

def plot_shap(nids, model_type):
	if model_type == 'ALLO':
		n_mat_features = 6
		txt_n_features = 11
	elif model_type == 'EGO':
		n_mat_features = 8
		txt_n_features = 8
	else:
		raise

	model_cells = []
	model_features = []
	j = 0

	mat = np.zeros((len(nids), n_mat_features))
	for i, n in enumerate(nids):
		print(n)
		d = shelve.open(f"cache/{n}/shap")
		ks = list(d.keys())
		if len(ks) == 0: continue
		if len(ks) == 2:
			features = eval(ks[0].split("|")[1])
			n_features = len(features)
			model_key = ks[0]
			if n_features != txt_n_features:
				features = eval(ks[1].split("|")[1])
				n_features = len(features)
				model_key = ks[1]
			if n_features != txt_n_features:
				continue
		#print(features)
		if model_key not in d: continue
		shap_vals = d[model_key][0]
		#pprint.pprint(shap_vals)
		#print(sparsity(shap_vals.values()))
		vals = list(shap_vals.values())
		#print(vals)
		if len(vals) == n_mat_features:
			mat[j] = vals
			j += 1
			model_cells.append(n)
			model_features = shap_vals.keys()

	plot_mat = mat[:j]
	fig, ax = plt.subplots(1)

	fig.suptitle(model_type,x=0.1)
	ax.matshow(plot_mat, vmin=0)
	ax.set_yticks(range(len(model_cells)))
	ax.set_yticklabels(model_cells)

	ax.set_xticks(range(len(model_features)))
	ax.set_xticklabels(map(lambda x: x.replace("_F_","_"), model_features), rotation=70, fontsize=6)
	

plot_shap(nids, "ALLO")
plot_shap(nids, "EGO")
plt.show()
