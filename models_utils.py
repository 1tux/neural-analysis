from typing import List
import pygam

def create_pos_smoother(x_idx, y_idx):
    return pygam.te(x_idx, y_idx, n_splines=[10, 5], constraints='concave')

def create_distance_smoother(idx):
    return pygam.s(idx, n_splines=10)

# TODO: fix pygam library to support circular constraints!
def create_angle_smoother(idx):
    return pygam.s(idx, n_splines=10)
    return pygam.s(idx, n_splines=10, constraints='circular')

def build_formula(covariates):
    features_subset_copy = covariates.copy()

    formula = []
    two_d_idx = []
    one_d_idx = []
    features = []
    real_idx = 0
    for idx, f in enumerate(covariates):
        bat_id = f.split("_")[1]
        suffix = f.split("_")[-1]
        if f.endswith("X"):
            y_idx = covariates.index(f[:-1]+"Y")
            features_subset_copy.remove(f)
            features_subset_copy.remove(f[:-1]+"Y")
            formula.append(create_pos_smoother(idx, y_idx))
            # features.append(f"BAT_{bat_id}_F_POS")
            features.append((f"BAT_{bat_id}_F_X", f"BAT_{bat_id}_F_Y"))
            two_d_idx.append(real_idx)
            real_idx += 1
        elif not f.endswith("Y"):
            one_d_idx.append(real_idx)
            real_idx += 1
            if f.endswith("_D"):
                formula.append(create_distance_smoother(idx))
                features.append(f"BAT_{bat_id}_F_D")
            if f.endswith("_A") or f.endswith("HD"):
                formula.append(create_angle_smoother(idx))
                features.append(f"BAT_{bat_id}_F_{suffix}")
            if f.endswith("_Dp"):
                formula.append(create_distance_smoother(idx))
                features.append(f"PAIR_{bat_id}_F_Dp")

    if len(formula) == 1:
        formula = formula[0]
    else:
        formula = sum(formula[1:], formula[0]) # converts list of elements to sum
    return features, formula

def features_to_covarietes(features):
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