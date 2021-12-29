import sys
import pandas as pd

from state import State
from results import Results1
import data_loaders
import preprocessing
import models_lib
import plot_lib
import store_results

def pipeline1(neuron_id: int):
    # handles paths, supports raw data, simulated_data, csv, matlab...
    data = data_loaders.Loader2()(neuron_id)

    # remove nans, scaling, feature-engineering
    data = preprocessing.Preprocess1()(data)

    results = Results1()
    results.maps = preprocessing.maps(data)

    # setup models with some hyper-params
    sub_models = [
    models_lib.AlloModel(),
    # models_lib.AlloModel(covariates=['BAT_0_F_HD', 'BAT_0_F_X', 'BAT_0_F_Y']),
    models_lib.EgoModel(covariates=['BAT_1_F_A', 'BAT_1_F_D']),
    # models_lib.PairModel()
    ]
    best_model = models_lib.get_best_model(sub_models, data)
    print("Top model:", type(best_model).__name__)

    results.models = sub_models
    # results.shap = best_model.shapley()
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
    main([72])
    # main(sys.argv[1:])