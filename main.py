#!/usr/bin/env python3
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import mean_poisson_deviance, mean_squared_error, mean_tweedie_deviance
from sklearn.model_selection import train_test_split, GroupKFold, TimeSeriesSplit, ShuffleSplit
from sklearn.model_selection import KFold
import functools
import shelve
import copy
import dataclasses
from scipy.stats import pearsonr, spearmanr
import os

import math
import matplotlib.pyplot as plt
import binascii
import base64

from conf import Conf
import data_manager
import models
import store_results
import rate_maps
import model_maps
import features_lib
import models_utils
import dic

@dataclasses.dataclass
class Results:
    def store(self):
        pass

def cross_validated_train_model(neuron_id: int, model: models.Model, dataprop: data_manager.DataProp, shuffle_index: int = 0) -> models.ModelledNeuron:
    data = dataprop.data
    y = data[features_lib.get_label_name()]
    X = data[model.covariates]
    cache = {} if not Conf().USE_CACHE else shelve.open(Conf().CACHE_FOLDER + "models")

    modelled_neuron = models.ModelledNeuron(model, neuron_id, shuffle_index)
    n_groups = len(X) // Conf().TIME_BASED_GROUP_SPLIT
    groups = range(n_groups)

    def group_index_to_timeframe(group_indices):
        group_size = Conf().TIME_BASED_GROUP_SPLIT
        return sum([list(range(group_index * group_size, (group_index+1) * group_size)) for group_index in group_indices], [])

    cv = KFold(n_splits=10, random_state=1337, shuffle=True)
    # cv = KFold(n_splits=10, shuffle=False)
    all_preds = pd.Series(np.ones(len(X)) * 0.5)

    final_model_coef_sum = None
    i = 0
    for train_group_indices, test_group_indices in cv.split(groups):
        train_index = group_index_to_timeframe(train_group_indices)
        test_index = group_index_to_timeframe(test_group_indices)
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]

        print(f"Training...{i+1}/{cv.n_splits}")

        model.train_model(X_train, y_train)
        
        if final_model_coef_sum is None:
            final_model_coef_sum = model.gam_model.coef_ 
        final_model_coef_sum += model.gam_model.coef_ 

        y_pred = model.gam_model.predict(X_test)
        all_preds.loc[test_index] = y_pred
        i += 1
    final_model_coef_sum /= cv.n_splits 

    modelled_neuron.model = model
    modelled_neuron.model.X_test = X
    modelled_neuron.model.y_pred = all_preds
    modelled_neuron.model.gam_model.coef_ = final_model_coef_sum
    model.score = dic.calc_dic(modelled_neuron, 10)

    res = modelled_neuron
    return res

def train_model(neuron_id: int, model: models.Model, dataprop: data_manager.DataProp, shuffle_index: int = 0) -> models.ModelledNeuron:
    '''
        splits data using TimeSeriesSplit, chunks are based on autocorrealtion analysis -> to train/test.
        shuffles if neccessary.
        train a model and evalulate.
        uses cache to save up time.

    '''
    data = dataprop.data
    y = data[features_lib.get_label_name()]
    X = data[model.covariates]
    cache = {} if not Conf().USE_CACHE else shelve.open(Conf().CACHE_FOLDER + "models")

    modelled_neuron = models.ModelledNeuron(model, neuron_id, shuffle_index)
    cache_key = modelled_neuron.get_key()
    if cache_key not in cache:
        train_idx, test_idx = list(ShuffleSplit(n_splits = 1, test_size=0.05, random_state=1337).split(X))[0]
        groups = list(np.array(range(0, len(y))) // Conf().TIME_BASED_GROUP_SPLIT)
        train_idx, test_idx = list(GroupKFold(n_splits = 2).split(X, y, groups))[0]
        X_train, X_test, y_train, y_test = X.loc[train_idx], X.loc[test_idx], y[train_idx], y[test_idx]
        
        # 5 seconds chunks + cross-validation 90/10 of these chunks
        # take the best model for further analysis...
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

        # print("Splitting...")
        # groups = list(np.array(range(0, len(y))) // Conf().TIME_BASED_GROUP_SPLIT)
        # gen_groups = GroupKFold(n_splits=2).split(X, y, groups)
        # ## gen_groups = TimeSeriesSplit(gap=Conf().TIME_BASED_GROUP_SPLIT, max_train_size=None, n_splits=2, test_size=None).split(X, y, groups)
        # for g in gen_groups:
        #     train_index, test_index = g
        #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
        # print("Splitted!")


        X_train, y_train, X_test, y_test = X, y, X, y

        #print("Training...")
        model.train_model(X_train, y_train)
        #print("Predicting...")
        model.y_pred = model.gam_model.predict(X_test)
        model.X_test = X_test
        model.y_test = y_test
        #model.X_train = X_train
        #model.y_train = y_train
        #cache[cache_key] = modelled_neuron
        model.score = dic.calc_dic(modelled_neuron, n_samples=10)
        modelled_neuron.model = model
        res = modelled_neuron
        # cache[cache_key] = modelled_neuron

    # something to do with shelve caching engine... 
    # TODO: delete this. it is also important before we want to override cache...   
    #obj = cache[cache_key]
    #if 'X_test' not in dir(model):
    #    obj.model.X_test = X
    #    obj.model.y_test = y
    #obj.model.score = dic.calc_dic(obj, 10)
    #cache[cache_key] = obj
    #res = cache[cache_key]
    #if type(cache) != dict:
    #    cache.close()
    return res

def model_key_to_output_file(m: ModelledNeuron):
    key_no_features = m.get_key().split("|")
    features_subset = key_no_features.pop(1).strip('][').replace("\'","").split(', ')
    full_features_list = sorted(m.model.build_covariates_list())
    binary_vec = ''

    for f in full_features_list:
        if f in features_subset:
            binary_vec += '1'
        else:
            binary_vec += '0'
            
    key_no_features.insert(2, binary_vec)
    key_no_features.insert(0, key_no_features.pop(1))
    key_no_features = "_".join(key_no_features)
    # output_name = base64.b64encode(m.get_key().encode('utf-8')).decode('utf-8')[::2]
    output_name = key_no_features
    if Conf().CROSS_VALIDATE:
        output_name += "_CV"
    return output_name

def calc_maps_and_plot_models(dataprop: data_manager.DataProp, data_maps: List[rate_maps.RateMap], sub_models: List[models.ModelledNeuron]):
    for m in sub_models:

        # train_idx, test_idx = list(ShuffleSplit(n_splits = 1, test_size=0.2, random_state=1337).split(dataprop.data[m.model.covariates]))[0]
        # test_idx = train_idx
        test_idx = m.model.X_test.index
        model_fr_map = model_maps.ModelFiringRate(dataprop, m.model, m.shuffle_index, test_idx)
        my_model_maps = model_maps.build_maps(m.model, data_maps)

        filter_width = Conf().TIME_BASED_GROUP_SPLIT
        spikes_count = dataprop.orig_spikes_count
        smooth_fr = np.convolve(spikes_count, [1] * filter_width, mode='same') / filter_width
        smooth_fr_no_nans = pd.Series(smooth_fr).loc[dataprop.no_nans_indices]
        #smooth_fr_no_nans_shuffled = np.roll(smooth_fr_no_nans, -m.shuffle_index)
        #smooth_fr_no_nans_shuffled = smooth_fr_no_nans_shuffled[test_idx]
        shuffle_indices = (dataprop.no_nans_indices[test_idx] + m.shuffle_index) % np.max(dataprop.no_nans_indices[test_idx])
        # fr_map = rate_maps.FiringRate(smooth_fr_no_nans_shuffled, shuffle_indices)
        # print(dataprop.data.index)
        fr_map = rate_maps.FiringRate(smooth_fr_no_nans, shuffle_indices)
        fr_map = rate_maps.FiringRate(smooth_fr_no_nans, smooth_fr_no_nans.reset_index().index)
        fr_map = rate_maps.FiringRate(smooth_fr_no_nans, dataprop.no_nans_indices / Conf().FRAME_RATE / 60)
        fr_map.process()
        mpd = mean_poisson_deviance(fr_map.map_, model_fr_map.y + np.finfo(float).eps)
        pearson_correlation = pearsonr(fr_map.y, model_fr_map.y)[0]
        spearman_correlation = spearmanr(fr_map.y, model_fr_map.y)[0]
        mse = mean_squared_error(fr_map.y, model_fr_map.y)
        m.model.R = pearson_correlation
        # print("Mean Poisson Deviance of the model:", mpd)
        # print("deviance score", dev)
        # print("R", pearson_correlation)
        
        stats_text = "Stats:\n"
        stats_text += "MSE: " + str(mse) + "\n"
        stats_text += "R: " + str(pearson_correlation) + "\n"
        stats_text += "R_spearman: " + str(spearman_correlation) + "\n"
        stats_text += "MPD: " + str(mpd) + "\n"
        stats_text += "pDIC: " + str(m.model.score[0]) + "\n"
        stats_text += "DIC: " + str(m.model.score[1]) + "\n"

        stats_text += "timepoints: " + str(len(dataprop.no_nans_indices)) + "\n"
        stats_text += "no.spikes: " + str(dataprop.spikes_count.sum()) + "\n"
        stats_text += "loglikelihood-per spike: " + str(m.model.gam_model.statistics_["loglikelihood"] / dataprop.spikes_count.sum()) + "\n"
        for c in ["loglikelihood", "edof", "AIC", "AICc", "UBRE"]:
            stats_text += f"{c}: {m.model.gam_model.statistics_[c]}\n"
        print(stats_text)
        output_name = model_key_to_output_file(m)
        open(f"{Conf().STATS_PATH}/{output_name}.txt", 'w').write(stats_text)
        open(f"{Conf().STATS_PATH}/{output_name}.json", 'w').write(stats_text)

            m.model.plot(dataprop.n_bats, fr_map, model_fr_map, data_maps, my_model_maps, stats_text)
            # print(model.gam_model.summary())
            # print(m.model.gam_model.logs_['deviance'])

        figure = plt.gcf()
        figure.set_size_inches(24, 10)
        output_name = model_key_to_output_file(m)
        if Conf().COSYNE:
            plt.savefig(f'{Conf().IMGS_PATH}{output_name}.svg', bbox_inches='tight', dpi=300) 
            plt.savefig(f'{Conf().IMGS_PATH}{output_name}.pdf', bbox_inches='tight', dpi=300)    
        plt.savefig(f'{Conf().IMGS_PATH}{output_name}.png', bbox_inches='tight', dpi=100)

        if Conf().TO_PLOT:
            plt.show()
        else:
            plt.close()
        

def models_stats_text(sub_models: List[models.ModelledNeuron]):
    out = ""
    for m in sub_models:
        for c in ["loglikelihood", "edof", "AIC", "AICc", "UBRE"]:
            out += f"{c}: {m.model.gam_model.statistics_[c]}\n"
        out += f"pR^2: {m.model.gam_model.statistics_['pseudo_r2']['explained_deviance']}"
    return out

def pipeline1(neuron_id: int):
    # handles paths, supports raw data, simulated_data, csv, matlab...
    print(f"Loading Data [{neuron_id}]...")
    try:
        data = data_manager.Loader8()(neuron_id, Conf().SHUFFLE_INDEX)
    except FileNotFoundError:
        print("File not found")
        raise
        return

    print("Loaded!")

    # remove nans, scaling, feature-engineering
    dataprop = data_manager.DataProp1(data)
    if dataprop.spikes_count.sum() < 100:
        print("Too few spikes")
        return

    results = Results()
    results.dataprop = dataprop
    print("Building Data Maps...")
    results.data_maps = rate_maps.build_maps(dataprop)
    results.fr_map = rate_maps.FiringRate(dataprop.spikes_count, dataprop.no_nans_indices)
    print("Data Maps Built!")
    # setup models with some hyper-params
    sub_models = [
    models.AlloModel(n_bats=dataprop.n_bats, max_iter=20, fit_intercept=True),
    models.EgoModel(n_bats=dataprop.n_bats, max_iter=20, fit_intercept=True),
    ]
    print("Training and comparing Models....")
    if Conf().CROSS_VALIDATE:
        sub_models = [cross_validated_train_model(neuron_id, model, dataprop, Conf().SHUFFLE_INDEX) for model in sub_models]
    else:
        sub_models = [train_model(neuron_id, model, dataprop, Conf().SHUFFLE_INDEX) for model in sub_models]
    best_model1 = min(sub_models, key=lambda i:i.model.score[1])
    best_model2 = min(sub_models, key=lambda i:i.model.gam_model.statistics_['AIC'])

    if len(sub_models) != 2:
        delta_DIC = 0
        print("Warning no delta_DIC calculation")
    else:
        delta_DIC = sub_models[0].model.score[1] - sub_models[1].model.score[1]

    if best_model1.model.__class__ == best_model2.model.__class__ and abs(delta_DIC) >= 10:
        best_model = best_model1
        print("CLS:", best_model1.model.__class__,best_model2.model.__class__)
    else:
        print("No statistical difference. delta_DIC:", delta_DIC)
        best_model = best_model1
        # Conf().RUN_SHAPLEY = False
    print("Top model:", type(best_model.model).__name__)

    # run shuffles
    # if Conf().SHUFFLES:
    #    start = Conf().SHUFFLES_MIN_GAP
    #    end = len(dataprop.no_nans_indices) - Conf().SHUFFLES_MIN_GAP
    #    shuffles = []
    #    shift_size = math.ceil((end - start) / Conf().SHUFFLES_JMPS)
    #    assert end > 0, "not enough timepoints for this gap"
    #    print("SHUFFLES START, END, SHIFT_SIZE:", start, end, shift_size)
    #    for shift in range(start, end, shift_size):
    #        if len(shuffles) == Conf().SHUFFLES:
    #            break
    #        print(shift)
    #        shuffles.append(train_model(neuron_id, copy.deepcopy(best_model.model), dataprop, shift))

    #    calc_maps_and_plot_models(dataprop, results.data_maps, shuffles)
    #    for s in shuffles:
    #        print("SHUFFLE", s.model.R)
    # end of running shuffles

    results.models = sub_models
    calc_maps_and_plot_models(dataprop, results.data_maps, sub_models)
    sub_models_ = []
    for best_model in sub_models:
        if Conf().RUN_SHAPLEY:
            # results.shap = best_model.model.shapley(dataprop.data)
            sub_models_ = []
            def powerset(iterable):
                import itertools
                "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
                s = list(iterable)
                return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))

            subsets = powerset(best_model.model.features)
            best_model_cls = type(best_model.model)
            for subset in subsets:
                if subset != ():
                    covariates = features_lib.features_to_covariates(subset)[:]
                    if best_model_cls == models.AlloModel:
                        new_model = models.AlloModel(covariates=covariates)
                    elif best_model_cls == models.EgoModel:
                        new_model = models.EgoModel(covariates=covariates)
                    else:
                        exit(-1)
                    new_model.n_bats = best_model.model.n_bats
                    sub_models_.append(new_model)
            print("RUNNING:", len(sub_models_), "models")

            sub_models_ = [train_model(neuron_id, model, dataprop) for model in sub_models_]
            for model in sub_models_:
                model.model.n_bats = best_model.model.n_bats # fix models that were stored wrongly in the cache.

        if len(sub_models_):
            calc_maps_and_plot_models(dataprop, results.data_maps, sub_models_)

            shap_res = {}
            for m in sub_models_:
                features_subset = tuple(sorted([f.name for f in m.model.features]))
                print(features_subset)
                shap_res[features_subset] = m.model.gam_model.statistics_['pseudo_r2']['explained_deviance']
            shapley = models_utils.calc_shapley_values(shap_res)
            print("Shapley for best model:")
            for s in shapley[0].items():   
                print(s[0], f"{s[1]:.3f}")
        
            # store shapley in a dedicated file
            f = shelve.open(Conf().CACHE_FOLDER + "shap")
            f[best_model.get_key()] = shapley

            if Conf().FEATURE_SELECTION:
                reduced_model = min(sub_models_, key=lambda model: model.model.gam_model.statistics_['AIC'])
                # Conf().TO_PLOT=True
                calc_maps_and_plot_models(dataprop, results.data_maps, [reduced_model])
                # Conf().TO_PLOT=False


    dataprop.store()
    # best_model.model.store()
    results.store()

    return results

# handle args
# overwrite state
# execute model over arguments
# chooses one of the implemented pipelines to execute
def main(args):
    args = list(map(lambda s: s.lower(), args))
    print(args)
    Conf().nid = int(args[0])
    Conf().SHUFFLE_INDEX = 0 # the default is no shuffling...
    if "no-cache" in args:
        Conf().USE_CACHE = False
    if "no-plot" in args:
        Conf().TO_PLOT = False
    if "shuffle" in args:
        shuffle_index = args[args.index("shuffle")+1]
        Conf().SHUFFLE_INDEX = int(shuffle_index)
    if "shapley" in args:
        Conf().RUN_SHAPLEY = True
    if "cache-path" in args:
        Conf().CACHE_FOLDER = args[args.index("cache-path")+1] + "/"
        if not os.path.exists(Conf().CACHE_FOLDER): os.mkdir(Conf().CACHE_FOLDER)
    if "cosyne" in args:
        Conf().COSYNE = True
    else:
        Conf().COSYNE = False
    if "xval" in args:
        Conf().CROSS_VALIDATE = True
    else:
        Conf().CROSS_VALIDATE = False

    Conf().nid = int(args[0])
    pipeline1(Conf().nid)

if __name__ == "__main__":
    if len(sys.argv) == 1: sys.argv.append(72)
    main(sys.argv[1:])
