import pandas as pd
import numpy as np
import h5py
from conf import Conf
from typing import Optional, List
from dataclasses import dataclass

class DataProp:
    '''
        Data + Properties that were captured on the data.
        For example, the number of bats, or the relevant net we studying.
    '''
    def __init__(self):
        pass

class DataProp1(DataProp):
    def __init__(self, data, net="net1"):
        # handle dataset
        
        # orig_data will remain, data will be changed
        self.orig_data = self.data = data
        self.bats_names = self.extract_bats_names()
        self.n_bats = len(self.bats_names)
        self.net_name = net
        self.net_dims = Conf().DIMS_PER_NET[net]
        self.prepcocess()

        # handle label
        self.spikes_count = self.data["neuron"]
        self.orig_spikes_count = self.orig_data["neuron"]
        self.calc_firing_rate()

        # handle features
        self.one_d_features = self.get_one_d_features()
        self.two_d_features = self.get_two_d_features()

    def prepcocess(self):
        self.remove_nans()
        self.add_pairwise_features()
        
    def remove_nans(self):
        no_nans = self.data.dropna()
        self.no_nans_indices = no_nans.index
        self.data = no_nans.reset_index(drop=True)

    # TODO: implement add_pairwise_features()
    def add_pairwise_features(self):
        pass

    def calc_firing_rate(self, filter_width=120) -> np.array:
        self.firing_rate = np.convolve(self.spikes_count, [1] * filter_width, mode='same') / filter_width
        return self.firing_rate    

    def get_one_d_features(self):
        features = [self.get_feature_name(0, "HD")]
        for bat_name in range(1, self.n_bats):
            features.append(self.get_feature_name(bat_name, "A"))
            features.append(self.get_feature_name(bat_name, "D"))
        return features

    def get_two_d_features(self) -> List[str]:
        features = []
        for bat_name in range(self.n_bats):
            two_d_feature = (self.get_feature_name(bat_name, "X"), self.get_feature_name(bat_name, "Y"))
            features.append(two_d_feature)
        return features

    def get_feature_name(self, bat_name, feature_name):
        assert str(bat_name) in self.bats_names
        return f"BAT_{bat_name}_F_{feature_name}"

    def extract_bats_names(self) -> List[str]:
        bats_names = pd.Series(self.data.columns.str.extract('BAT_(\d)_*', expand=False).unique()).dropna().values
        return bats_names

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