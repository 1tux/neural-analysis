from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GroupKFold, TimeSeriesSplit
import functools

from conf import Conf
from results import Results
import data_manager
import models
import plot_lib
import store_results
import rate_maps
import model_maps
import features_lib

def get_best_model(sub_models: List[models.Model], data: pd.DataFrame) -> models.Model:
    y = data[features_lib.get_label_name()]
    groups = list(np.array(range(0, len(y))) // Conf().TIME_BASED_GROUP_SPLIT)
    for model in sub_models:
        X = data[model.covariates]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1337)

        print("Splitting...")
        gen_groups = GroupKFold(n_splits=2).split(X, y, groups)
        gen_groups = TimeSeriesSplit(gap=250, max_train_size=None, n_splits=2, test_size=None).split(X, y, groups)
        for g in gen_groups:
            train_index, test_index = g
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
        print("Splitted!")

        print("Training...")
        model.train_model(X_train, y_train)
        print("Predicting...")
        model.y_pred = model.gam_model.predict(X)
        model.evaulate()

    best_model = max(sub_models, key=lambda i:i.score)
    return best_model

def plot_models(dataprop, data_maps, fr_map, sub_models: List[models.Model]):
    for model in sub_models:
        model_fr_map = model_maps.ModelFiringRate(dataprop, model)
        my_model_maps = model_maps.build_maps(model, data_maps)
        model.plot(dataprop.n_bats, fr_map, model_fr_map, data_maps, my_model_maps)
        r2 = r2_score(fr_map.map_, model_fr_map.y)
        print("R^2 of the model:", r2)
        model.gam_model.summary()

def pipeline1(neuron_id: int):
    # cache_CACHE_FOLDER + "nid.pkl"
    # handles paths, supports raw data, simulated_data, csv, matlab...
    print("Loading Data...")
    data = data_manager.Loader2()(neuron_id)
    print("Loaded!")

    # remove nans, scaling, feature-engineering
    dataprop = data_manager.DataProp1(data)

    results = Results()
    results.dataprop = dataprop
    # TODO: split firing-rate map, to a differnet function.
    print("Building Data Maps...")
    results.data_maps = rate_maps.build_maps(dataprop)
    results.fr_map = rate_maps.FiringRate(dataprop)
    print("Data Maps Built!")
    # setup models with some hyper-params
    sub_models = [
    # models.AlloModel(n_bats=dataprop.n_bats, max_iter=20, fit_intercept=False),
    models.AlloModel(covariates=['BAT_0_F_HD', 'BAT_1_F_X', 'BAT_1_F_Y'], max_iter=20),
    # models.EgoModel(n_bats=dataprop.n_bats, max_iter=30),
    models.EgoModel(covariates=['BAT_2_F_A', 'BAT_2_F_D'], max_iter=20),
    # models.PairModel()
    ]
    print("Training and comparing Models....")
    best_model = get_best_model(sub_models, dataprop.data)
    print("Top model:", type(best_model).__name__)

    results.models = sub_models
    # results.shap = best_model.shapley()
    # results.models_maps = model_maps.build_maps(best_model, results.data_maps)
    # results.model_fr_map = model_maps.ModelFiringRate(dataprop, best_model)
    results.shuffles = best_model.run_shuffles()

    plot_models(dataprop, results.data_maps, results.fr_map, sub_models)

    if 'shap' in dir(results):
        print("SHAP for best model:")
        for s in results.shap[0].items():   
            print(s[0].name, f"{s[1]:.3f}")

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
    main([61])
    # main(sys.argv[1:])