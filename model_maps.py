import models
import pandas as pd
import numpy as np

def build_maps(dataprop, model):
    maps = {}
    for term_id, term in enumerate(model.features):
        maps[term] = build_model_map(model, term, term_id)
    return maps

def build_model_map(model, term, term_id):
    term_type = term.split("_")[-1]
    if term_type == "POS":
        return ModelMap2D(model, term, term_id)
    if term_type in ["HD", "A", "D"]:
        return ModelMap1D(model, term, term_id)

class ModelMap:
    '''
        gets a model and feature name
        generate pos, angular, distance maps
    '''
    def __init__(self, model, term, term_id):
        self.model = model
        self.term_id = term_id
        self.term = term
        self.feature_type = term.split("_")[-1]
        self.XX = None
        self.map_ = None
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
        pass 

    def plot(self, ax):
        pass