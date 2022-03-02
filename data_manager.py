import pandas as pd
import numpy as np
import h5py
import typing
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
import os.path

from conf import Conf
import features_lib

def neuron_id_to_day(neuron_id):
    if 1 <= neuron_id <= 29: return "d191222"
    if 31 <= neuron_id <= 53: return "d191221"
    if 56 <= neuron_id <= 80: return "d191220"
    if 82 <= neuron_id <= 104: return "d191223"
    if 106 <= neuron_id <= 129: return "d191224"
    if 132 <= neuron_id <= 144: return "d191225"
    if 145 <= neuron_id <= 161: return "d191226"
    if 163 <= neuron_id <= 174: return "d191229"
    if 176 <= neuron_id <= 190: return "d191231"
    if 191 <= neuron_id <= 207: return "d200101"
    if 208 <= neuron_id <= 216: return "d200102"
    if 217 <= neuron_id <= 227: return "d200108"
    if 228 <= neuron_id <= 233: return "d190603"
    if 236 <= neuron_id <= 244: return "d190604"
    if 246 <= neuron_id <= 252: return "d190610"
    if 254 <= neuron_id <= 265: return "d190612"
    if 270 <= neuron_id <= 272: return "d190617"
    if 274 <= neuron_id <= 280: return "d190924"
    if 282 <= neuron_id <= 291: return "d190925"
    if 292 <= neuron_id <= 298: return "d190926"
    if 300 <= neuron_id <= 339: return "d190928"
    if 375 <= neuron_id <= 377: return "d200419"
    if 378 <= neuron_id <= 380: return "d200420"
    if 381 <= neuron_id <= 386: return "d200421"
    if 387 <= neuron_id <= 389: return "d200422"
    if 390 <= neuron_id <= 392: return "d200423"
    if 393 <= neuron_id <= 394: return "d200425"
    if 395 <= neuron_id <= 400: return "d200426"
    if 401 <= neuron_id <= 406: return "d200427"
    if 407 <= neuron_id <= 415: return "d200428"
    if 416 <= neuron_id <= 423: return "d200429"
    if 424 <= neuron_id <= 428: return "d200430"

class DataProp:
    '''
        Data + Properties that were captured on the data.
        For example, the number of bats, or the relevant net we studying.
    '''
    def __call__(self):
        pass  

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
        def calc_firing_rate(spikes_count, filter_width) -> np.array:
            return np.convolve(spikes_count, [1] * filter_width, mode='same') / filter_width 
        self.orig_firing_rate = calc_firing_rate(self.orig_spikes_count, Conf().TIME_BASED_GROUP_SPLIT)
        self.firing_rate = self.orig_firing_rate[self.spikes_count.index]

        self.covariates = self.data.drop(columns=[features_lib.get_label_name()]).columns.to_list()
        self.features = features_lib.covariates_to_features(self.covariates)

    def split_train_test():
        X = self.data[self.data.columns.difference(features_lib.get_label_name())]
        y = self.data[features_lib.get_label_name()]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.8, random_state=1337)

        # print("Splitting...")
        # gen_groups = GroupKFold(n_splits=2).split(X, y, groups)
        # ## gen_groups = TimeSeriesSplit(gap=Conf().TIME_BASED_GROUP_SPLIT, max_train_size=None, n_splits=2, test_size=None).split(X, y, groups)
        # for g in gen_groups:
        #     train_index, test_index = g
        #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
        # print("Splitted!")

    def prepcocess(self):
        self.remove_nans()
        self.add_pairwise_features()


    def store(self):
        pass

class DataLoader:
    '''
        Handles paths and different file formats
    '''
    def __call__(self):
        pass

class Loader1(DataLoader):
    def __call__(self, nid: int) -> pd.DataFrame:
        df = pd.read_csv(r"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\b2305_d191220_simplified_behaviour.csv").drop(columns=['Unnamed: 0'])
        df['neuron'] = pd.read_csv(r"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\72_b2305_d191220.csv")['0']
        return df

class Loader2(DataLoader):
    def __call__(self, nid: int) -> pd.DataFrame:
        day = neuron_id_to_day(nid)
        df = pd.read_csv(fr"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\b2305_{day}_simplified_behaviour.csv").drop(columns=['Unnamed: 0'])
        # neuron_path = fr"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\{nid}_b2305_{day}_cell_analysis.mat"
        neuron_path = fr"C:\tmp\20210506 - neurons\{nid}_b2305_{day}_cell_analysis.mat"
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

class Loader4(DataLoader):
    def __call__(self, nid: int) -> pd.DataFrame:
        day = neuron_id_to_day(nid)
        df = pd.read_csv(fr"C:\Users\itayy.WISMAIN\data\MSc\behavioral\b2305_{day}_simplified_behaviour.csv").drop(columns=['Unnamed: 0'])
        neuron_path = fr"C:\Users\itayy.WISMAIN\data\MSc\neural\{nid}_b2305_{day}_cell_analysis.mat"
        d = h5py.File(neuron_path, "r")
        spikes = np.array(d['cell_analysis']['spikes_per_frame']).T[0]
        df['neuron'] = spikes
        return df

class Loader5(DataLoader):
    def __call__(self, nid: int) -> pd.DataFrame:
        day = neuron_id_to_day(nid)
        df = pd.read_csv(os.path.join(Conf().INPUT_FOLDER, fr"b2305_{day}_simplified_behaviour.csv")).drop(columns=['Unnamed: 0'])
        neuron_path = os.path.join(Conf().INPUT_FOLDER, fr"{nid}_b2305_{day}_cell_analysis.mat")
        d = h5py.File(neuron_path, "r")
        spikes = np.array(d['cell_analysis']['spikes_per_frame']).T[0]
        df['neuron'] = spikes
        return df

class Loader6(DataLoader):
    def __call__(self, nid: int) -> pd.DataFrame:
        if nid < 1000:
            return Loader4()(nid)

        day = f"d1912{20 + (nid // 1000 - 1)}"
        import glob
        neuron_path_ = os.path.join(Conf().INPUT_FOLDER, "simulated", day)
        simulated_list = list(map(os.path.basename, glob.glob(neuron_path_ + "/*")))

        file_name = simulated_list[(nid % 1000) % len(simulated_list)]
        print(nid, nid % len(simulated_list), day, file_name)
        behavioral_path = os.path.join(Conf().INPUT_FOLDER, fr"b2305_{day}_simplified_behaviour.csv")
        neuron_path = os.path.join(neuron_path_, file_name)

        df = pd.read_csv(behavioral_path).drop(columns=['Unnamed: 0']) 
        spikes = pd.read_csv(neuron_path)['0']
        df['neuron'] = spikes
        return df

class Loader7(DataLoader):
    def __call__(self, nid: int) -> pd.DataFrame:
        if nid < 1000:
            day = neuron_id_to_day(nid)
            import glob
            path = glob.glob(fr"C:\Users\itayy.WISMAIN\git\neural-analysis\inputs\Behavior\all behaviour 20220209\*_{day}_simplified_behaviour.csv")[0]
            rec_bat = os.path.basename(path).split('_')[0]
            df = pd.read_csv(path).drop(columns=['Unnamed: 0'])
            neuron_path = fr"Z:\for_Itay\Cells\20220301 - neurons\{nid}_{rec_bat}_{day}_cell_analysis.mat"
            d = h5py.File(neuron_path, "r")
            spikes = np.array(d['cell_analysis']['spikes_per_frame']).T[0]
            df['neuron'] = spikes
            return df

        day = f"d1912{20 + (nid // 1000 - 1)}"
        import glob
        neuron_path_ = os.path.join(Conf().INPUT_FOLDER, "simulated", day)
        simulated_list = list(map(os.path.basename, glob.glob(neuron_path_ + "/*")))

        file_name = simulated_list[(nid % 1000) % len(simulated_list)]
        print(nid, nid % len(simulated_list), day, file_name)
        behavioral_path = os.path.join(Conf().INPUT_FOLDER, fr"b2305_{day}_simplified_behaviour.csv")
        neuron_path = os.path.join(neuron_path_, file_name)

        df = pd.read_csv(behavioral_path).drop(columns=['Unnamed: 0']) 
        spikes = pd.read_csv(neuron_path)['0']
        df['neuron'] = spikes
        return df