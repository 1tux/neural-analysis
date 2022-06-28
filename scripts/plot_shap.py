import shelve
import matplotlib.pyplot as plt
import pprint

def plot(nid, model_type, otuput_path):
	assert model_type in ["Allo", "Ego"]

	shap = shelve.open(f"X:/neural-analysis/outputs/cache/{nid}/shap")
	vals = shap[[k for k in shap.keys() if k.startswith(model_type)][0]][0]
	pprint.pprint(vals)

	vals = dict(sorted(vals.items(), key=lambda item: item[1]))
	feature_names = list(vals.keys())
	feature_names = [x.replace("_A"," Angle").replace("_D", " Distance").replace("_F", "").replace("BAT_","BAT ") for x in feature_names]
	feature_vals = list(vals.values())


	fig, ax = plt.subplots()
	plt.suptitle("Variable Importance", size=20)
	plt.barh(feature_names, feature_vals)
	ax.tick_params(axis='both', which='major', labelsize=16)
	plt.tight_layout()
	plt.savefig(otuput_path + fr"neuron{nid}_{model_type}_shapley.png")

if __name__ == '__main__':
	plot(72, 'Allo', r"C:\\tmp\\")