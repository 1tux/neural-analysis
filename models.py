from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
import pygam
import time
import itertools
import math
import typing
import models_utils

from conf import Conf

class Model:
    def __init__(self, covariates: typing.Optional[list[str]] = None, use_cache:bool=True, n_bats = typing.Optional[int]):
        # position comprised of two covarietes: x,y
        self.n_bats = n_bats
        self.covariates = covariates or self.build_covariates_list()

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_trained = False

        self.features = None # position is a single 2-d feature
        self.formula = None
        self.train_test_ratio = Conf().TRAIN_TEST_RATIO
        self.gam_model = None
        self.score = None
        self.use_cache = use_cache

    # TODO: change to time-series split
    # TODO: this is unrelated to model
    def split_train_test(self, data):
        self.X = data[self.covariates]
        self.y = data['neuron']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=1337)

    def train_model(self, data: pd.DataFrame):
        start_time = time.time()
        self.data = data
        self.split_train_test(data)
        self.features, self.formula = models_utils.build_formula(self.covariates)
        self.gam_model = pygam.PoissonGAM(self.formula, max_iter=25)
        self.gam_model.fit(self.X_train, self.y_train)
        self.is_trained = True
        print(f"training {(type(self).__name__)} in {time.time() - start_time:2f} seconds")
        # self.gam_model.summary()
        # print(self.gam_model.statistics_)


    # TODO: implement DIC / WAIC to account for model's complexity
    def evaulate(self):
        assert self.is_trained
        self.score = self.gam_model.statistics_['loglikelihood']

    # retrain sub-models over all the covariets
    # estimate shapley values
    def shapley(self):
        results = train_submodels(self.__class__, self.features, self.data)
        shapley = calc_shapley_values(results)
        print(results)
        print(shapley)

    def generate_maps(self):
        pass

    def run_shuffles(self):
        pass

    def build_covariates_list(self):
        print("are you mad?!")
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
        covariates = ["BAT_0_F_HD"]
        for i in range(self.n_bats):
            covariates.append(f"BAT_{i}_F_X")
            covariates.append(f"BAT_{i}_F_Y")

        return covariates

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
        covariates = []
        for i in range(1, self.n_bats):
            covariates.append(f"BAT_{i}_F_A")
            covariates.append(f"BAT_{i}_F_D")
        return covariates