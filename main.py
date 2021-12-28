import sys
import pandas as pd

import state
import data_loaders
import preprocessing
import models_lib
import postprocessing # not sure ?
import plot_lib
import store_results

def pipeline1(neuron_id: int):
	# handles paths, supports raw data, simulated_data, csv, matlab...
	data = data_loaders.Loader1()(neuron_id)
	state.get_state().n_bats = preprocessing.get_number_of_bats(data)

	# remove nans, scaling, feature-engineering split to sub_models
	data = preprocessing.Preprocess1()(data)

	# setup models with some hyper-params
	sub_models = [
	models_lib.AlloModel(),
	models_lib.EgoModel(),
	# models_lib.PairModel()
	]
	best_model = models_lib.get_best_model(sub_models, data)

	results = postprocessing.Results1()
	results.models = sub_models
	results.maps = preprocessing.maps(data)
	results.shap = best_model.shapley()
	results.models_maps = best_model.generate_maps()
	results.shuffles = best_model.run_shuffles()

	results.plot()
	results.store()

# handle args
# overwrite state
# execute model over arguments
# chooses one of the implemented pipelines to execute
def main(args):
	nid = int(args[0])
	pipeline1(nid)

if __name__ == "__main__":
	main([1000])
	# main(sys.argv[1:])