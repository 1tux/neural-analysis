import sys
import pandas as pd
from typing import List
from sklearn.metrics import r2_score
from conf import Conf
from results import Results1

import data_manager
import models
import plot_lib
import store_results
import rate_maps
import model_maps

def get_best_model(sub_models: List[models.Model], data: pd.DataFrame) -> models.Model:
    for model in sub_models:
        model.train_model(data)
        model.evaulate()

    best_model = max(sub_models, key=lambda i:i.score)
    return best_model

def memoize(f):
    memory = {}
    def inner(num):
        if num not in memory:         
            memory[num] = f(num)
        return memory[num]
  
    return inner

@memoize
def pipeline1(neuron_id: int):
    # cache_CACHE_FOLDER + "nid.pkl"
    # handles paths, supports raw data, simulated_data, csv, matlab...
    data = data_manager.Loader2()(neuron_id)

    # remove nans, scaling, feature-engineering
    dataprop = data_manager.DataProp1(data)

    results = Results1()
    results.dataprop = dataprop
    # TODO: split firing-rate map, to a differnet function.
    results.data_maps = rate_maps.build_maps(dataprop)
    results.fr_map = rate_maps.FiringRate(dataprop)

    # setup models with some hyper-params
    sub_models = [
    # models.AlloModel(n_bats=dataprop.n_bats, max_iter=30, fit_intercept=False),
    # models.AlloModel(covariates=['BAT_0_F_HD', 'BAT_0_F_X', 'BAT_0_F_Y']),
    models.EgoModel(n_bats=dataprop.n_bats),
    # models.EgoModel(covariates=['BAT_1_F_A', 'BAT_1_F_D']),
    # models.PairModel()
    ]
    best_model = get_best_model(sub_models, dataprop.data)
    print("Top model:", type(best_model).__name__)

    results.models = sub_models
    results.best_model = best_model
    # results.shap = best_model.shapley()
    results.models_maps = model_maps.build_maps(best_model, results.data_maps)
    results.model_fr_map = model_maps.ModelFiringRate(dataprop, best_model)
    results.shuffles = best_model.run_shuffles()

    results.r2 = r2_score(results.fr_map.map_, results.model_fr_map.y)
    print(results.r2)

    results.plot()

    dataprop.store()
    best_model.store()
    results.store()
    return results

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