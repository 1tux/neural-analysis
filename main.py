import sys
import pandas as pd
from typing import List

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

    # best model is the one with the highest score
    best_model = sorted(sub_models, key=lambda i:i.score)[-1]
    return best_model

def pipeline1(neuron_id: int):
    # handles paths, supports raw data, simulated_data, csv, matlab...
    data = data_manager.Loader2()(neuron_id)

    # remove nans, scaling, feature-engineering
    dataprop = data_manager.DataProp1(data)

    results = Results1()
    results.dataprop = dataprop
    results.rate_maps = rate_maps.build_maps(dataprop)

    # setup models with some hyper-params
    sub_models = [
    models.AlloModel(n_bats=dataprop.n_bats),
    # models.AlloModel(covariates=['BAT_0_F_HD', 'BAT_0_F_X', 'BAT_0_F_Y']),
    #models.EgoModel(covariates=['BAT_1_F_A', 'BAT_1_F_D']),
    # models.PairModel()
    ]
    best_model = get_best_model(sub_models, dataprop.data)
    print("Top model:", type(best_model).__name__)

    results.models = sub_models
    # results.shap = best_model.shapley()
    results.models_maps = model_maps.build_maps(dataprop, best_model)
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