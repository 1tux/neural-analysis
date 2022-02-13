import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pygam
import time
import typing
import matplotlib.pyplot as plt
import shelve 
import dataclasses

import models_utils
import features_lib
from conf import Conf
import model_maps
import rate_maps


class Model:
    def __init__(self, covariates: typing.Optional[list[str]] = None, use_cache:bool=True, n_bats = typing.Optional[int], **kwargs):
        # position comprised of two covarietes: x,y
        self.n_bats = n_bats
        self.covariates = covariates or self.build_covariates_list()
        self.features = features_lib.covariates_to_features(self.covariates)

        # self.y_test = None
        self.y_pred = None
        self.is_trained = False
        # self.shuffle_index = 0
        # self.neuron_id = None

        self.formula = None
        self.train_test_ratio = Conf().TRAIN_TEST_RATIO
        self.gam_model = None
        self.score = None
        self.use_cache = use_cache
        self.kwargs = kwargs

    def store(self):
        pass

    def plot(self, **kwargs):
        pass

    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        # TODO: try loading from cache before traning...
        start_time = time.time()
        self.formula = models_utils.build_formula(self.features)
        self.gam_model = pygam.PoissonGAM(self.formula, **self.kwargs)
        self.gam_model.fit(X_train, y_train)

        self.is_trained = True
        print(f"training {(type(self).__name__)}:{self.covariates} in {time.time() - start_time:2f} seconds")
        # self.gam_model.summary()
        # print(self.gam_model.statistics_)


    # retrain sub-models over all the covariets
    # estimate shapley values
    def shapley(self):
        results = models_utils.train_submodels(self.__class__, self.features, self.data, self.kwargs)
        shapley = models_utils.calc_shapley_values(results)
        return shapley

    def build_covariates_list(self):
        pass

@dataclasses.dataclass
class ModelledNeuron:
    model: Model
    neuron_id: int
    shuffle_index: int

    def get_key(self):
        '''
            gets a model and deduce unique key for it to be extract and stored in cache.
        '''
        model_type_str = self.model.__class__.__name__
        covariates_str = str(sorted(self.model.covariates))
        nid_str = str(self.neuron_id)
        shuffle_index_str = str(self.shuffle_index)
        return "|".join([model_type_str, covariates_str, nid_str, shuffle_index_str])


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
        covariates = [features_lib.get_feature_name(0, "HD")]
        for i in range(self.n_bats):
            covariates.append(features_lib.get_feature_name(i, "X"))
            covariates.append(features_lib.get_feature_name(i, "Y"))

        return covariates

    def plot(self, n_bats: int, data_fr_map: rate_maps.FiringRate, model_fr_map: model_maps.ModelFiringRate,\
    data_maps: typing.List[rate_maps.RateMap], model_maps: typing.List[model_maps.ModelMap], stats: str):

        fig = plt.figure()
        grid = plt.GridSpec(5, n_bats, wspace=1, hspace=1)

        pos_maps = list(filter(lambda m: m.type_ == features_lib.FeatureType.POS, data_maps))
        model_pos_maps = list(filter(lambda m: m.type_ == features_lib.FeatureType.POS, model_maps))
        # assert len(pos_maps) == n_bats, (len(pos_maps), n_bats)
        # assert len(model_pos_maps) == n_bats, (len(model_pos_maps), n_bats)

        hd_maps = list(filter(lambda m: m.type_ == features_lib.FeatureType.HD, data_maps))
        hd_model_maps = list(filter(lambda m: m.type_ == features_lib.FeatureType.HD, model_maps))
        # assert len(hd_maps) == 1
        # assert len(hd_model_maps) == 1

        for i, m in enumerate(pos_maps):
            pos_axis = fig.add_subplot(grid[1, i])
            data_maps[m].plot(pos_axis)

        for i, m in enumerate(model_pos_maps):
            pos_axis = fig.add_subplot(grid[2, i])
            model_maps[m].plot(pos_axis)

        hd_axis = fig.add_subplot(grid[3, 0])
        data_maps[hd_maps[0]].plot(hd_axis)
        model_maps[hd_model_maps[0]].plot(hd_axis)

        fr_axis = fig.add_subplot(grid[0, :n_bats])
        data_fr_map.plot(fr_axis)
        model_fr_map.plot(fr_axis)

        x_ticks = fr_axis.get_xticks().tolist()[:-1]
        x_tickslabels = (np.array(x_ticks) / Conf().FRAME_RATE // 60).astype('int')
        fr_axis.set_xticks(x_ticks) # to ignore some weird warning
        fr_axis.set_xticklabels(x_tickslabels)
        fr_axis.set_xlabel("Minutes")
        fr_axis.set_ylabel("Spikes/second")

        stats_axis = fig.add_subplot(grid[4, :n_bats])
        stat_lines = stats.splitlines()
        first_half = "\n".join(stat_lines[:(1+len(stat_lines)) // 2])
        second_half = "\n".join(stat_lines[(1+len(stat_lines)) // 2:])
        stats_axis.text(0.5, 0, first_half)
        stats_axis.text(0.8, 0, second_half)
        stats_axis.axis('off')

        plt.show()

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
            covariates.append(features_lib.get_feature_name(i, "A"))
            covariates.append(features_lib.get_feature_name(i, "D"))
        return covariates

    def plot(self, n_bats: int, data_fr_map: rate_maps.FiringRate, model_fr_map: model_maps.ModelFiringRate,\
    data_maps: typing.List[rate_maps.RateMap], model_maps: typing.List[model_maps.ModelMap], stats: str):

        fig = plt.figure()
        grid = plt.GridSpec(4, n_bats-1, wspace=0.4, hspace=1)

        angle_maps = list(filter(lambda m: m.type_ == features_lib.FeatureType.A, data_maps))
        model_angle_maps = list(filter(lambda m: m.type_ == features_lib.FeatureType.A, model_maps))
        # assert len(angle_maps) == n_bats - 1, (len(angle_maps), n_bats)
        # assert len(model_angle_maps) == n_bats - 1, (len(model_angle_maps), n_bats)

        distance_maps = list(filter(lambda m: m.type_ == features_lib.FeatureType.D, data_maps))
        distance_model_maps = list(filter(lambda m: m.type_ == features_lib.FeatureType.D, model_maps))
        # assert len(distance_maps) == n_bats - 1, (len(distance_maps), n_bats)
        # assert len(distance_model_maps) == n_bats -1, (len(distance_model_maps), n_bats)

        for i, (m, m2) in enumerate(zip(angle_maps, model_angle_maps)):
            angle_axis = fig.add_subplot(grid[1, i])
            data_maps[m2].plot(angle_axis)
            model_maps[m2].plot(angle_axis)

        for i, (m, m2) in enumerate(zip(distance_maps, distance_model_maps)):
            distance_axis = fig.add_subplot(grid[2, i])
            data_maps[m2].plot(distance_axis)
            model_maps[m2].plot(distance_axis)

        fr_axis = fig.add_subplot(grid[0, :n_bats])
        data_fr_map.plot(fr_axis)
        model_fr_map.plot(fr_axis)

        x_ticks = fr_axis.get_xticks().tolist()[:-1]
        # x_ticks = data_fr_map.x[::len(data_fr_map.x) // 10]
        x_tickslabels = (np.array(x_ticks) // Conf().FRAME_RATE // 60).astype('int') # every 25 minutes?
        fr_axis.set_xticks(x_ticks) # to ignore some weird warning
        fr_axis.set_xticklabels(x_tickslabels)
        fr_axis.set_xlabel("Minutes")
        fr_axis.set_ylabel("Spikes/second")

        stats_axis = fig.add_subplot(grid[3, :n_bats])
        stat_lines = stats.splitlines()
        first_half = "\n".join(stat_lines[:(1+len(stat_lines)) // 2])
        second_half = "\n".join(stat_lines[(1+len(stat_lines)) // 2:])
        stats_axis.text(0.5, 0, first_half)
        stats_axis.text(0.8, 0, second_half)
        stats_axis.axis('off')
        plt.show()