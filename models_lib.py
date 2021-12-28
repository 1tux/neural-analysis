from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
import pygam
import time

import state

class Model:
    def __init__(self, covariates = None, use_cache=True):
        # position comprised of two covarietes: x,y
        if covariates is None:
            self.build_covariates_list()
        else:
            self.covariates = covariates

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_trained = False

        self.features = None # position is a single 2-d feature
        self.formula = None
        self.train_test_ratio = state.get_state().train_test_ratio
        self.n_bats = state.get_state().n_bats
        self.gam_model = None
        self.score = None
        self.use_cache = use_cache

    # TODO: change to time-series split
    def split_train_test(self, data):
        self.X = data[self.covariates]
        self.y = data['neuron']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=1337)


    def train_model(self, data: pd.DataFrame):
        start_time = time.time()
        self.split_train_test(data)
        self.build_formula()
        self.gam_model = pygam.PoissonGAM(self.formula, max_iter=5)
        self.gam_model.fit(self.X_train, self.y_train)
        self.is_trained = True
        print(f"training {(type(self).__name__)} in {time.time() - start_time:2f} seconds")
        self.gam_model.summary()
        # print(self.gam_model.statistics_)

    def build_formula(self):
        features_subset_copy = self.covariates.copy()

        formula = []
        two_d_idx = []
        one_d_idx = []
        cols = []
        real_idx = 0
        for idx, f in enumerate(self.covariates):
            bat_id = f.split("_")[1]
            suffix = f.split("_")[-1]
            if f.endswith("X"):
                y_idx = self.covariates.index(f[:-1]+"Y")
                features_subset_copy.remove(f)
                features_subset_copy.remove(f[:-1]+"Y")
                formula.append(create_pos_smoother(idx, y_idx))
                cols.append(f"BAT_{bat_id}_POS")
                two_d_idx.append(real_idx)
                real_idx += 1
            elif not f.endswith("Y"):
                one_d_idx.append(real_idx)
                real_idx += 1
                if f.endswith("_D"):
                    formula.append(create_distance_smoother(idx))
                    cols.append(f"BAT_{bat_id}_D")
                if f.endswith("_A") or f.endswith("HD"):
                    formula.append(create_angle_smoother(idx))
                    cols.append(f"BAT_{bat_id}_{suffix}")
                if f.endswith("_Dp"):
                    formula.append(create_distance_smoother(idx))
                    cols.append(f"PAIR_{bat_id}_Dp")

        if len(formula) == 1:
            formula = formula[0]
        else:
            formula = sum(formula[1:], formula[0]) # converts list of elements to sum

        self.formula = formula
        self.features = cols


    # TODO: implement DIC / WAIC to account for model's complexity
    def evaulate(self):
        assert self.is_trained
        self.score = self.gam_model.statistics_['loglikelihood']

    # retrain sub-models over all the covariets
    # estimate shapley values
    def shapley(self):
        pass

    def generate_maps(self):
        pass

    def run_shuffles(self):
        pass

    def build_covariates_list(self):
        pass

class AlloModel(Model):
    def build_covariates_list(self):
        """
        example in case there are 5 bats
        [
                'BAT_0_F_X', 'BAT_0_F_Y', 'BAT_0_F_HD',
                'BAT_1_F_X', 'BAT_1_F_Y',
                'BAT_2_F_X', 'BAT_2_F_Y',
                'BAT_3_F_X', 'BAT_3_F_Y',
                'BAT_4_F_X', 'BAT_4_F_Y'
        ]"""
        self.covariates = ["BAT_0_F_HD"]
        for i in range(state.get_state().n_bats):
            self.covariates.append(f"BAT_{i}_F_X")
            self.covariates.append(f"BAT_{i}_F_Y")

class EgoModel(Model):
    def build_covariates_list(self):
        """
        example in case there are 5 bats
        [
                'BAT_1_F_D', 'BAT_1_F_A',
                'BAT_2_F_D', 'BAT_2_F_A',
                'BAT_3_F_D', 'BAT_3_F_A',
                'BAT_4_F_D', 'BAT_4_F_A'
        ]"""
        self.covariates = []
        for i in range(1, state.get_state().n_bats):
            self.covariates.append(f"BAT_{i}_F_A")
            self.covariates.append(f"BAT_{i}_F_D")

def get_best_model(sub_models: list[models_lib.Model], data: pd.DataFrame) -> models_lib.Model:
    scores = []
    for model in sub_models:
        model.train_model(data)
        model.evaulate()
        scores.append((model.score, model))

    # best model is the one with the highest score
    best_model = sorted(scores, key=lambda i:i[0])[-1][1]
    return best_model

def create_pos_smoother(x_idx, y_idx):
    return pygam.te(x_idx, y_idx, n_splines=[10, 5], constraints='concave')

def create_distance_smoother(idx):
    return pygam.s(idx, n_splines=10)

# TODO: fix pygam library to support circular constraints!
def create_angle_smoother(idx):
    return pygam.s(idx, n_splines=10)
    return pygam.s(idx, n_splines=10, constraints='circular')