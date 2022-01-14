from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GroupKFold, TimeSeriesSplit
import functools
import shelve
import copy

from conf import Conf
from results import Results
import data_manager
import models
import store_results
import rate_maps
import model_maps
import features_lib

def get_key_per_model(model, shuffle_index):
    model_type_str = model.__class__.__name__
    covariates_str = str(sorted(model.covariates))
    nid_str = str(nid)
    shuffle_index_str = str(shuffle_index)
    return "|".join([model_type_str, covariates_str, nid_str, shuffle_index_str])

def train_model(model, data, shuffle_index=0):
    y = data[features_lib.get_label_name()]
    if shuffle_index:
        y = np.roll(y, shuffle_index)
    groups = list(np.array(range(0, len(y))) // Conf().TIME_BASED_GROUP_SPLIT)
    X = data[model.covariates]
    d = shelve.open(Conf().CACHE_FOLDER + "models")
    key = get_key_per_model(model, shuffle_index)
    if key not in d:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1337)

        print("Splitting...")
        gen_groups = GroupKFold(n_splits=2).split(X, y, groups)
        gen_groups = TimeSeriesSplit(gap=Conf().TIME_BASED_GROUP_SPLIT, max_train_size=None, n_splits=2, test_size=None).split(X, y, groups)
        for g in gen_groups:
            train_index, test_index = g
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # break
        print("Splitted!")

        print("Training...")
        model.train_model(X_train, y_train)
        print("Predicting...")
        model.y_pred = model.gam_model.predict(X)
        model.evaulate()
        d[key] = model
    return d[key]

def plot_models(dataprop, data_maps, fr_map, sub_models: List[models.Model]):
    for model in sub_models:
        model_fr_map = model_maps.ModelFiringRate(dataprop, model)
        my_model_maps = model_maps.build_maps(model, data_maps)
        model.plot(dataprop.n_bats, fr_map, model_fr_map, data_maps, my_model_maps)
        r2 = r2_score(fr_map.map_, model_fr_map.y)
        print("R^2 of the model:", r2)
        # model.gam_model.summary()

def pipeline1(neuron_id: int):
    # cache_CACHE_FOLDER + "nid.pkl"
    # handles paths, supports raw data, simulated_data, csv, matlab...
    print("Loading Data...")
    data = data_manager.Loader5()(neuron_id)
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
    models.AlloModel(n_bats=dataprop.n_bats, max_iter=25, fit_intercept=False),
    # models.AlloModel(covariates=['BAT_0_F_HD', 'BAT_1_F_X', 'BAT_1_F_Y'], max_iter=20),
    # models.EgoModel(n_bats=dataprop.n_bats, max_iter=30),
    models.EgoModel(covariates=['BAT_2_F_A', 'BAT_2_F_D'], max_iter=25, fit_intercept=False),
    # models.PairModel()
    ]
    print("Training and comparing Models....")
    sub_models = [train_model(model, dataprop.data) for model in sub_models]
    best_model = max(sub_models, key=lambda i:i.score)
    print("Top model:", type(best_model).__name__)

    sub_models.append(train_model(copy.deepcopy(sub_models[1]), dataprop.data, 10000))

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
    global nid
    nid = int(args[0])
    pipeline1(nid)

if __name__ == "__main__":
    if len(sys.argv) == 1: sys.argv.append(72)
    main(sys.argv[1:])
