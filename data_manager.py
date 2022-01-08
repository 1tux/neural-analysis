import pandas as pd
import numpy as np
import h5py
import typing
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass

from conf import Conf
import features_lib

class DataProp:
    '''
        Data + Properties that were captured on the data.
        For example, the number of bats, or the relevant net we studying.
    '''
    def __init__(self):
        pass

    def calc_firing_rate(self, filter_width=120) -> np.array:
        self.firing_rate = np.convolve(self.spikes_count, [1] * filter_width, mode='same') / filter_width
        # self.firing_rate = gaussian_filter1d(self.spikes_count, 30)
        return self.firing_rate    

    def remove_nans(self):
        no_nans = self.data.dropna()
        self.no_nans_indices = no_nans.index
        self.data = no_nans.reset_index(drop=True)

    # TODO: implement add_pairwise_features()
    def add_pairwise_features(self):
        pass


class DataProp1(DataProp):
    def __init__(self, data, net="net1"):
        # handle dataset
        
        # orig_data will remain, data will be changed
        self.orig_data = self.data = data
        self.bats_names = features_lib.extract_bats_names(self.data)
        self.n_bats = len(self.bats_names)
        self.net_name = net
        self.net_dims = Conf().DIMS_PER_NET[net]
        self.prepcocess()

        # handle label
        self.spikes_count = self.data[features_lib.get_label_name()]
        self.orig_spikes_count = self.orig_data[features_lib.get_label_name()]
        self.calc_firing_rate()

        self.covariates = self.data.drop(columns=[features_lib.get_label_name()]).columns.to_list()
        self.features = features_lib.covariates_to_features(self.covariates)

    def prepcocess(self):
        self.remove_nans()
        self.add_pairwise_features()

    def store(self):
        pass

class DataLoader:
    '''
        Handles paths and different file formats
    '''
    def __init__(self):
        pass

class Loader1(DataLoader):
    def __call__(self, nid: int) -> pd.DataFrame:
        df = pd.read_csv(r"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\b2305_d191220_simplified_behaviour.csv").drop(columns=['Unnamed: 0'])
        df['neuron'] = pd.read_csv(r"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\72_b2305_d191220.csv")['0']
        return df

class Loader2(DataLoader):
    def __call__(self, nid: int) -> pd.DataFrame:
        df = pd.read_csv(r"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\b2305_d191220_simplified_behaviour.csv").drop(columns=['Unnamed: 0'])
        neuron_path = r"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\72_b2305_d191220_cell_analysis.mat"
        d = h5py.File(neuron_path, "r")
        spikes = np.array(d['cell_analysis']['spikes_per_frame']).T[0]
        df['neuron'] = spikes
        return df

class Loader3(DataLoader):
    def __call__(self, nid: int) -> pd.DataFrame:
        df = pd.read_csv(r"C:\Users\itayy\Documents\Bat-Lab\data\behavioral_data\parsed\b2305_d191220_simplified_behaviour.csv").drop(columns=['Unnamed: 0'])
        neuron_path = r"Z:\for_Itay\20210506 - neurons\72_b2305_d191220_cell_analysis.mat"
        d = h5py.File(neuron_path, "r")
        spikes = np.array(d['cell_analysis']['spikes_per_frame']).T[0]
        df['neuron'] = spikes
        return df