import numpy as np
import scipy
from data_manager import DataProp
from typing import Tuple, List, Optional, Union
from conf import Conf
import features_lib

def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def build_maps(dataprop):
    maps = {}
    for f in dataprop.features:
        if f.dim() == 1:
            maps[f] = RateMap1D(dataprop, f)
        if f.dim() == 2:
            maps[f] = RateMap2D(dataprop, f)

    return maps

class RateMap:
    '''
        loads data
        process data and create a map
        plot map
    '''
    def __init__(self, dataprop, feature: Optional[features_lib.Feature] = None):
        self.dataprop = dataprop
        self.feature = feature
        self.map_ = None
        self.axis = None
        self.frame_rate = Conf().FRAME_RATE
        self.spikes_count = dataprop.spikes_count
        self.process()

    def process(self):
        pass

    def plot(self, ax):
        pass

class FiringRate(RateMap):
    def __init__(self, dataprop):
        super().__init__(dataprop)
        
    def process(self):
        self.x = self.dataprop.no_nans_indices
        self.map_ = self.y = self.frame_rate * self.dataprop.firing_rate

    def plot(self, ax):
        ax.plot(self.x, self.y, '.', markersize=1, alpha=0.5, label='test-firing-rates')

class RateMap1D(RateMap):
    def __init__(self, dataprop, feature: features_lib.Feature):
        self.bin_size = Conf().ONE_D_PLOT_BIN_SIZE
        self.time_spent_threshold = Conf().ONE_D_TIME_SPENT_THRESHOLD
        super().__init__(dataprop, feature)

    def process(self):
        feature_value = self.dataprop.data[self.feature.covariates[0]]
        self.map_, self.axis = calculate_1d_ratemap(feature_value,\
        self.spikes_count, self.frame_rate, self.bin_size, self.time_spent_threshold)
        self.mean_fr = np.nanmean(self.map_)
        # print(self.mean_fr, np.nanmean(self.dataprop.spikes_count), np.nanmean(self.dataprop.orig_spikes_count))
        return self.map_

    def plot(self, ax):
        if ax is None: return
        ax.plot(self.axis[:-1], self.map_)
        # ax.set_title(f"FR: MAX={np.nanmax(result):.2f} Hz Mean={np.nanmean(result):.2f} Hz")
        # ax.set_ylim(bottom=0)
        if self.feature.type_ in [features_lib.FeatureType.A, features_lib.FeatureType.HD]:
            ax.set_xticks(np.arange(0, 360, 60))
        peak_fr = np.nanmax(self.map_)
        ax.set_title(f"{self.feature.name} - peak_fr {peak_fr:.3f} Hz")
        return np.nanmean(self.map_)

class RateMap2D(RateMap):
    def __init__(self, dataprop, feature: features_lib.Feature):
        self.bin_size = Conf().TWO_D_PLOT_BIN_SIZE
        self.time_spent_threshold = Conf().TWO_D_TIME_SPENT_THRESHOLD
        self.filter_size = Conf().GAUSSIAN_FILTER_SIZE
        self.filter_sigma = Conf().GAUSSIAN_FILTER_SIGMA
        self.cutoff = Conf().TWO_D_PRECENTILE_CUTOFF
        super().__init__(dataprop, feature)

    def process(self):
        x_pos = self.dataprop.data[self.feature.covariates[0]]
        y_pos = self.dataprop.data[self.feature.covariates[1]]
        self.width, self.height = self.dataprop.net_dims

        self.map_, self.not_enough_time_spent = calculate_pos_ratemap(x_pos, y_pos, self.spikes_count,\
        self.frame_rate, self.width, self.height, self.bin_size,\
        self.time_spent_threshold, self.filter_size, self.filter_sigma)
        self.mean_fr = np.nanmean(self.map_)
        # print(self.mean_fr, np.nanmean(self.dataprop.spikes_count), np.nanmean(self.dataprop.orig_spikes_count))
        return self.map_

    def plot(self, ax):
        if ax is None: return
        min_ = 0
        max_ = np.nanquantile(self.map_, self.cutoff)
        bin_size = self.bin_size

        x_plot_range = np.linspace(0, self.width // bin_size - bin_size / 2 + 1, 3)
        y_plot_range = np.linspace(0, self.height // bin_size - bin_size / 2 + 1, 3)

        ax.set_xticks(x_plot_range)
        ax.set_yticks(y_plot_range)

        ax.set_xticklabels((x_plot_range * bin_size + bin_size / 2).round(1))
        ax.set_yticklabels((y_plot_range * bin_size + bin_size / 2).round(1))

        img = ax.imshow(self.map_.T, cmap='jet', vmin=min_, vmax=max_)
        ax.set_title(self.feature.name)

def calculate_1d_ratemap(feature_value, spike_count,\
    frame_rate, bin_size, time_spent_threshold):
    min_x_value = np.floor(np.min(feature_value))
    max_x_value = np.ceil(np.max(feature_value))
    x_axis = np.linspace(min_x_value, max_x_value, bin_size)

    time_spent = np.histogram(feature_value, bins=x_axis)[0]

    rate_map = np.histogram(feature_value, bins=x_axis, weights=spike_count)[0].astype('float') / time_spent
    rate_map[time_spent < time_spent_threshold] = np.nan

    return frame_rate * rate_map, x_axis

def calculate_pos_ratemap(x_pos, y_pos, spike_counts, \
    frame_rate, width, height, \
    bin_size, time_spent_threshold, filter_size, filter_sigma):
    time_spent = np.histogram2d(x_pos, y_pos, [width // bin_size, height // bin_size], range=[(0, width), (0, height)])[0]
    time_spent = time_spent * (time_spent >= time_spent_threshold)

    spikes = np.histogram2d(x_pos, y_pos, [width // bin_size, height // bin_size], weights=spike_counts, range=[(0, width), (0, height)])[0]
    spikes2 = spikes * (time_spent >= time_spent_threshold)

    gauss_filter = fspecial_gauss(filter_size, filter_sigma)

    smooth_spikes = scipy.ndimage.correlate(spikes, gauss_filter, mode='constant')
    smooth_time_spent = scipy.ndimage.correlate(time_spent, gauss_filter, mode='constant')

    # the devision returns nans for 0/0 and inf for v/0 if v!=0. later on we replace the infs, and other places with nans
    old_err_setting = np.seterr(divide='ignore', invalid='ignore')
    smoothed_result = smooth_spikes / smooth_time_spent
    np.seterr(**old_err_setting)

    not_enought_time_spent = time_spent < time_spent_threshold
    smoothed_result[not_enought_time_spent] = np.nan


    return frame_rate * smoothed_result, not_enought_time_spent