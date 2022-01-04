
import pandas as pd
import numpy as np

from conf import Conf
import models
import features_lib

def build_maps(dataprop, model):
    maps = {}
    for term_id, term in enumerate(model.features):
        maps[term] = build_model_map(dataprop, model, term, term_id)

    maps["fr"] = ModelFiringRate(dataprop, model, "_fr", None)
    maps["fr"].feature_type = "fr"
    return maps

def build_model_map(dataprop, model, term, term_id):
    if term.dim() == 1:
        return ModelMap1D(dataprop, model, term, term_id)
    elif term.dim() == 2:
        return ModelMap2D(dataprop, model, term, term_id)

class ModelMap:
    '''
        gets a model and feature name
        generate pos, angular, distance maps
    '''
    def __init__(self, dataprop, model, term, term_id):
        self.dataprop = dataprop
        self.model = model
        self.term_id = term_id
        self.feature = term
        self.XX = None
        self.map_ = None
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
        ax.plot(XX, self.map_)

class ModelMap2D(ModelMap):
    def process(self):
        Xs = [np.linspace(0, 96, num=33).astype('int'), np.linspace(0, 45, num=16).astype('int')]
        self.XX = tuple(np.meshgrid(*Xs, indexing='ij'))
        self.map_ = np.exp(self.model.gam_model.partial_dependence(term=self.term_id, X=self.XX, meshgrid=True)).T

    def plot(self, ax):
        ax.clear()
        ax.imshow(self.map_, cmap='jet')

class ModelFiringRate(ModelMap):
    def process(self):
        self.x = self.dataprop.no_nans_indices
        self.map_ = self.y = self.model.gam_model.predict(self.model.X) * Conf().FRAME_RATE

    def plot(self, ax):
        ax.plot(self.x, self.y, '.', markersize=1, alpha=0.5, label='test-firing-rates')