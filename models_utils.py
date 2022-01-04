from typing import List
import functools
import operator

import features_lib
import pygam

def create_pos_smoother(x_idx, y_idx):
    return pygam.te(x_idx, y_idx, n_splines=[10, 5], constraints='concave')

def create_distance_smoother(idx):
    return pygam.s(idx, n_splines=10)

# TODO: fix pygam library to support circular constraints!
def create_angle_smoother(idx):
    return pygam.s(idx, n_splines=10)
    return pygam.s(idx, n_splines=10, constraints='circular')

def build_formula(features):
    # features = features_lib.covariates_to_features(covariates)
    feature_type_to_spline_map = {
        features_lib.FeatureType.HD: create_angle_smoother,
        features_lib.FeatureType.A: create_angle_smoother,
        features_lib.FeatureType.D: create_distance_smoother,
        features_lib.FeatureType.POS : create_pos_smoother,
    }
    feature_to_spline = lambda f: feature_type_to_spline_map[f.type_](*f.covariates_indices)
    formula = functools.reduce(operator.add, map(feature_to_spline, features))
    return formula

def features_to_covarietes(features):
    raise
    covariates = features[:]
    for i, f in enumerate(covariates):
        if "_POS" in f:
            covariates[i] = covariates[i].replace("_POS", "_Y")
            covariates.insert(i, covariates[i].replace("_Y", "_X"))
    return covariates

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))

def comb(n, k):
    return math.factorial(n) / math.factorial(k) / math.factorial(n - k)

def train_submodels(model, features, data):
    subsets = powerset(features)
    d = {}
    for subset in subsets:
        if subset != ():
            covariates = features_to_covarietes(list(subset))
            new_gam = model(covariates)
            new_gam.train_model(data)
            new_gam.evaulate()
            d[subset] = new_gam.gam_model.statistics_['pseudo_r2']['explained_deviance'] # new_gam.score
    return d

def calc_shapley_values(results):
    # input: results is a dictionary of tuples of features and their r^2 score (or any other target score).
    # the dictionary should contain the results of ALL subsets

    # output: a dictionary of shapley value per feature, scaled to 1.

    results[()] = 0  # not sure?
    features = sorted(list(results.keys())[::-1], key=lambda x: len(x))[-1]
    shapley = {}
    for f in features:
        v = 0
        for k in results:
            if f not in k: continue
            with_f = k
            without_f = k[:k.index(f)] + k[k.index(f) + 1:]
            scaling = comb(len(features) - 1, len(without_f))
            v += 1. / scaling * (results[with_f] - results[without_f])
        shapley[f] = v / len(features)

    shapley_scaled = {}
    for f in features:
        shapley_scaled[f] = shapley[f] / results[features] * 100

    sorted_shapley = sorted(list(shapley_scaled.keys()), key=lambda x: shapley_scaled[x])
    return shapley_scaled, sorted_shapley