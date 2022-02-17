from __future__ import annotations
import typing
import pandas as pd
import numpy as np
import operator
import functools

from conf import Conf
import features_lib
import rate_maps

class ModelMap:
    '''
        gets a model and feature name
        generate pos, angular, distance maps
    '''
    def __init__(self, model, feature_id, feature, rate_map):
        self.model = model
        self.term_id = feature_id
        self.feature = feature
        self.rate_map = rate_map
        self.XX = None
        self.map_ = None
        self.scaled_map = None
        self.feature_type = None
        self.process()

    def process(self):
        pass

    def plot(self, ax):
        pass

class ModelMap1D(ModelMap):
    def process(self):
        self.XX = self.model.gam_model.generate_X_grid(term=self.term_id)
        self.map_ = np.exp(self.model.gam_model.partial_dependence(term=self.term_id, X=self.XX))

    def plot(self, ax):
        tt = pd.Series(self.XX[-1] != 0) # wtf is this?
        XX = self.XX[:, tt[tt].index] # wtf is this?
        ax.plot(XX, self.scaled_map)

class ModelMap2D(ModelMap):
    def process(self):
        data_map_shape = self.rate_map.map_.shape
        data_not_enough_time_spent = self.rate_map.not_enough_time_spent
        # TODO: change these values based on nets sizes
        Xs = [np.linspace(0, 96, num=data_map_shape[0]).astype('int'), np.linspace(0, 45, num=data_map_shape[1]).astype('int')]
        self.XX = tuple(np.meshgrid(*Xs, indexing='ij'))
        self.map_ = self.model.gam_model.link.mu(self.model.gam_model.partial_dependence(term=self.term_id, X=self.XX, meshgrid=True), dist=None)
        self.align_maps()

    def align_maps(self):
        x_ticks = self.XX[0][:, 0]
        y_ticks = self.XX[1][0, :]
        bin_size = Conf().TWO_D_PLOT_BIN_SIZE
        width, height = Conf().DIMS_PER_NET["net1"]

        for x_idx, x in enumerate(x_ticks):
            for y_idx, y in enumerate(y_ticks):
                x_ = min(np.floor(x).astype('int') // bin_size, width // bin_size - 1)
                y_ = min(np.floor(y).astype('int') // bin_size, height // bin_size - 1)

                if self.rate_map.not_enough_time_spent[x_, y_]:
                    self.map_[x_idx, y_idx] = np.nan

    def plot(self, ax):
        ax.clear()
        ax.imshow(self.scaled_map.T, cmap='jet')
        peak_fr = np.nanmax(self.scaled_map)
        ax.set_title(f"peak_fr {peak_fr:.3f} Hz")

class ModelFiringRate:
    def __init__(self, dataprop, model, shuffle_index, test_idx):
        self.model = model
        self.x = (dataprop.no_nans_indices[test_idx]  + shuffle_index) % np.max(dataprop.no_nans_indices[test_idx])
        # self.x = (pd.Series(self.x).loc[test_idx])#.values
        # print("modelfr", self.x[:10])
        self.x = (dataprop.no_nans_indices[test_idx] + shuffle_index) % np.max(dataprop.no_nans_indices[test_idx])
        self.process()

    def process(self):
        self.y = self.map_ = self.model.y_pred * Conf().FRAME_RATE

    def plot(self, ax):
        ax.plot(self.x, self.y, '.', markersize=1, alpha=0.5, label='test-firing-rates')

def build_maps(model: models.Model, data_maps: rate_maps.RateMap) -> typing.List[ModelMap]:
    maps = {}
    for feature_id, feature in enumerate(model.features):
        if feature.dim() == 1:
            maps[feature] = ModelMap1D(model, feature_id, feature, data_maps[feature])
        elif feature.dim() == 2:
            maps[feature] = ModelMap2D(model, feature_id, feature, data_maps[feature])

    if model.gam_model.fit_intercept:
        intercept = model.gam_model.link.mu(model.gam_model.coef_[-1], dist=None)
    else:
        intercept = 1

    scale_maps(maps, intercept)
    return maps

def scale_maps(maps: Dict[ModelMap], intercept=1):
    keys = list(maps.keys())
    for key in keys:
        m = maps.pop(key)
        scale_map(m, maps, intercept)
        maps[key] = m

def scale_map(my_map: ModelMap, other_maps: Dict[ModelMap], intercept=1):
    maps_means = list(map(lambda k: np.nanmean(other_maps[k].map_), other_maps)) + [intercept]
    others_mean = functools.reduce(operator.mul, maps_means)
    my_map.scaled_map = my_map.map_ * others_mean * Conf().FRAME_RATE
    # print(my_map.feature.name, others_mean * Conf().FRAME_RATE)