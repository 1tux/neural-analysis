import sys
import pandas as pd

import state
import data_loaders
import preprocessing
import models_lib
import postprocessing # not sure ?
import plotlib
import store_results

def get_best_model(sub_models: list[models_lib.Model], sub_data: list[pd.DataFrame]) -> models_lib.Model:
	scores = []
	for model, data in zip(sub_models, sub_data):
		model.train_model(data)
		model.evaulate()
		scores.append((model.score, model))

	# best model is the one with the lowest score
	best_model = sorted(scores)[0][1]
	return best_model

def pipeline1(neuron_id: int):
	# handles paths, supports raw data, simulated_data, csv, matlab...
	data = data_loaders.Loader1(neuron_id)

	# remove nans, scaling, feature-engineering split to sub_models, deduce number of bats
	state.n_bats, data = preprocess.Preproccess1(data)

	# setup models with some hyper-params
	sub_models = [
	models_lib.AlloModel1(),
	models_lib.EgoModel1(),
	models_lib.PairModel1()
	]
	best_model = models_lib.get_best_model(sub_models, data)

	results = postprocessing.Results1()
	results.models = sub_models
	results.maps = preprocessing.maps(data)
	results.shap = best_model.shapely(best_model)
	results.models_maps = best_model.generate_maps()
	results.shuffles = best_model.run_shuffles()

	results.plot()
	results.store()

# handle args
# overwrite state
# execute model over arguments
# chooses one of the implemented pipelines to execute
def main(args):
	nid = int(args[1])
	pipeline1(nid)

if __name__ == "__main__":
	main(sys.argv)