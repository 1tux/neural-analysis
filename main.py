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
import dataclasses

from conf import Conf
import data_manager
import models
import store_results
import rate_maps
import model_maps
import features_lib
import models_utils

@dataclasses.dataclass
class Results:
    def store(self):
        pass

def train_model(neuron_id: int, model: models.Model, data: df.DataFrame, shuffle_index: int = 0):
    '''
        splits data using TimeSeriesSplit, chunks are based on autocorrealtion analysis -> to train/test.
        shuffles if neccessary.
        train a model and evalulate.
        uses cache to save up time.

    '''
    model.shuffle_index = shuffle_index
    model.neuron_id = neuron_id

    y = data[features_lib.get_label_name()]
    y = np.roll(y, shuffle_index)
    groups = list(np.array(range(0, len(y))) // Conf().TIME_BASED_GROUP_SPLIT)
    X = data[model.covariates]
    if Conf().USE_CACHE:
        cache = shelve.open(Conf().CACHE_FOLDER + "models")
    else:
        cache = {}
    cache_key = models_utils.get_key_per_model(model)
    if cache_key not in cache:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1337)

        print("Splitting...")
        gen_groups = GroupKFold(n_splits=2).split(X, y, groups)
        gen_groups = TimeSeriesSplit(gap=Conf().TIME_BASED_GROUP_SPLIT, max_train_size=None, n_splits=2, test_size=None).split(X, y, groups)
        for g in gen_groups:
            train_index, test_index = g
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
        print("Splitted!")

        print("Training...")
        model.train_model(X_train, y_train)
        print("Predicting...")
        model.y_pred = model.gam_model.predict(X)
        model.evaulate(X, y)
        cache[cache_key] = model
    cache[cache_key].evaulate(X, y)
    return cache[cache_key]

def plot_models(dataprop: data_manager.DataProp, data_maps: List[rate_maps.RateMap], sub_models: List[models.Model]):
    for model in sub_models:
        model_fr_map = model_maps.ModelFiringRate(dataprop, model)
        my_model_maps = model_maps.build_maps(model, data_maps)

        fr_map = rate_maps.FiringRate(np.roll(dataprop.spikes_count, model.shuffle_index), dataprop.no_nans_indices)
        fr_map.process()

        model.plot(dataprop.n_bats, fr_map, model_fr_map, data_maps, my_model_maps)
        r2 = r2_score(fr_map.map_, model_fr_map.y)
        print("R^2 of the model:", r2)
        # model.gam_model.summary()

def print_models_stats(sub_models: List[models.Model]):
    for model in sub_models:
        for c in ["loglikelihood", "edof", "AIC", "AICc", "UBRE"]:
            print(f"{c}: {model.gam_model.statistics_[c]}")
        print(f"pR^2: {model.gam_model.statistics_['pseudo_r2']['explained_deviance']}")

def pipeline1(neuron_id: int):
    # handles paths, supports raw data, simulated_data, csv, matlab...
    print("Loading Data...")
    data = data_manager.Loader4()(neuron_id)
    print("Loaded!")

    # remove nans, scaling, feature-engineering
    dataprop = data_manager.DataProp1(data)

    results = Results()
    results.dataprop = dataprop
    # TODO: split firing-rate map, to a differnet function.
    print("Building Data Maps...")
    results.data_maps = rate_maps.build_maps(dataprop)
    results.fr_map = rate_maps.FiringRate(dataprop.spikes_count, dataprop.no_nans_indices)
    print("Data Maps Built!")
    # setup models with some hyper-params
    sub_models = [
    models.AlloModel(n_bats=dataprop.n_bats, max_iter=25, fit_intercept=True),
    # models.AlloModel(covariates=['BAT_0_F_HD', 'BAT_1_F_X', 'BAT_1_F_Y'], max_iter=20),
    # models.EgoModel(n_bats=dataprop.n_bats, max_iter=30, fit_intercept=False),
    models.EgoModel(covariates=['BAT_2_F_A', 'BAT_2_F_D'], max_iter=25, fit_intercept=True),
    # models.PairModel()
    ]
    print("Training and comparing Models....")
    sub_models = [train_model(neuron_id, model, dataprop.data) for model in sub_models]
    best_model = max(sub_models, key=lambda i:i.score)
    print("Top model:", type(best_model).__name__)

    # run shuffles
    # TODO: recalculate data-firing-rate-map with shuffles!
    sub_models.append(train_model(neuron_id, copy.deepcopy(sub_models[-1]), dataprop.data, 10000))

    results.models = sub_models
    # results.shap = best_model.shapley()
    # results.models_maps = model_maps.build_maps(best_model, results.data_maps)
    # results.model_fr_map = model_maps.ModelFiringRate(dataprop, best_model)

    if Conf().TO_PLOT:
        plot_models(dataprop, results.data_maps, sub_models)
    print_models_stats(sub_models)

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
    Conf().nid = int(args[0])
    if "no-cache" in args:
        Conf().USE_CACHE = False
    if "no-plot" in args:
        Conf().TO_PLOT = False
    Conf().nid = int(args[0])
    pipeline1(Conf().nid)

if __name__ == "__main__":
    if len(sys.argv) == 1: sys.argv.append(72)
    main(sys.argv[1:])
