import sys
import numpy as np
import shelve
import copy

import models
from conf import Conf
import data_manager
import features_lib

def ll_per_sample(m, X, y, sample):
    m2 = copy.deepcopy(m)
    m2.gam_model.coef_ = sample
    return m2.gam_model.loglikelihood(X, y)


def calc_dic(model_architect, nid, n_samples=100):
    # load model from cache
    # load data

    k = models.ModelledNeuron(model_architect, nid, shuffle_index=0)
    d = shelve.open(Conf().CACHE_FOLDER + "models")

    if k.get_key() not in d:
        # print(f"Error! {nid} {model_architect.__class__} is untrained!")
        return (None, None)

    m = d[k.get_key()].model
    d2 = shelve.open(Conf().CACHE_FOLDER + "samples")

    to_add = False
    if k.get_key() in d2:
        samples_, lls_ = d2[k.get_key()]
        samples_ = samples_[:-1] # not including the mean_sample
        lls_ = lls_[:-1] # not including the mean_sample
        n_samples -= len(samples_)
        to_add = True

    if k.get_key() not in d2 or n_samples > 0:
        data = data_manager.Loader6()(nid)
        dataprop = data_manager.DataProp1(data)

        covariate_list = sorted(set(m.build_covariates_list()) & set(dataprop.data.columns))
        X = dataprop.data[covariate_list]
        y = dataprop.data[features_lib.get_label_name()]
        samples = m.gam_model.sample(X, y, quantity='coef', n_bootstraps=1, n_draws=n_samples)

        all_samples = samples
        if to_add:
            all_samples = np.append(samples_, samples, axis=0)
        mean_sample = np.mean(all_samples, axis=0).reshape(1, -1)
        samples = np.append(samples, mean_sample, axis=0) # adding the mean_sample to the end
        all_samples = np.append(all_samples, mean_sample, axis=0) # adding the mean_sample to the end

        lls = list(map(lambda s: ll_per_sample(m, X, y, s), samples))
        if to_add:
            lls = np.append(lls_, lls)
        d2[k.get_key()] = (all_samples, lls)

    samples, lls = d2[k.get_key()]
    mean_ll = lls[-1]
    pDIC = 2 * mean_ll - 2 * np.mean(lls[::-1]) # effective number of params
    dic = -2 * mean_ll + 2 * pDIC
    return pDIC, dic

def main(args):
    nid = int(args[0])
    n_samples = int(args[1]) if (len(args) > 1) else 100

    sub_models = [
        models.AlloModel(n_bats=5, max_iter=25, fit_intercept=True),
        models.EgoModel(n_bats=5, max_iter=25, fit_intercept=True)
    ]

    for m in sub_models:
        print(nid, m.__class__, calc_dic(m, nid, n_samples))

if __name__ == "__main__":
    if len(sys.argv) == 1: sys.argv.append(72)
    main(sys.argv[1:])
