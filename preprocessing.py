import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

import state

class Preprocess:
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

class Preprocess1(Preprocess):
    def __init__(self):
        pass

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return add_pairwise_features(remove_nans(data))

def remove_nans(data):
    print("Removing NANS")
    no_nans = data.dropna()
    state.get_state().no_nans_indices = no_nans.index
    return no_nans.reset_index(drop=True)

# TODO: implement add_pairwise_features()
def add_pairwise_features(data):
    return data

# TODO: implement add_pairwise_features()
def get_number_of_bats(data: pd.DataFrame) -> int:
    return 5

def spikes_to_firing_rate(spikes: np.array, filter_width=120) -> np.array:
    fr = (np.convolve(spikes, [1] * filter_width) / filter_width)[:len(spikes)]
    return fr

def maps(data):
    rate_map_ = rate_map(data, data['neuron'])
    fig, ax = plt.subplots()
    rate_map_plot2(rate_map_, ax)
    fig, ax = plt.subplots()
    one_d_map, x_axis = calc_1d_feature(data, data['neuron'], 0, "HD")
    plot_1d_feature(one_d_map, x_axis, "HD", ax)
    fig, ax = plt.subplots()
    plt.plot(state.get_state().no_nans_indices, state.get_state().FRAME_RATE * spikes_to_firing_rate(data['neuron']), '.', markersize=1, alpha=0.5, label='test-firing-rates')
    plt.show()


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def rate_map(df, neuron, bat_name=0, net="net1"):
    bin_size = state.get_state().TWO_D_PLOT_BIN_SIZE
    time_spent_threshold = state.get_state().TWO_D_TIME_SPENT_THRESHOLD
    width, height = state.get_state().dims_per_net[net]

    np.seterr(divide='ignore', invalid='ignore')

    x_plot_range = np.linspace(0, width // bin_size - bin_size / 2 + 1, 3).round(1)
    y_plot_range = np.linspace(0, height // bin_size - bin_size / 2 + 1, 3).round(1)

    bat_X = df[f"BAT_{bat_name}_F_X"]
    bat_Y = df[f"BAT_{bat_name}_F_Y"]

    time_spent = np.histogram2d(bat_X, bat_Y, [width // bin_size, height // bin_size], range=[(0, width), (0, height)])[0]
    time_spent = time_spent * (time_spent >= time_spent_threshold)

    spikes = np.histogram2d(bat_X, bat_Y, [width // bin_size, height // bin_size], weights=neuron, range=[(0, width), (0, height)])[0]
    spikes2 = spikes * (time_spent >= time_spent_threshold)

    gauss_filter = fspecial_gauss(state.get_state().GAUSSIAN_FILTER_SIZE, state.get_state().GAUSSIAN_FILTER_SIGMA)  # divides by 3, multiply by 4

    smooth_spikes = scipy.ndimage.correlate(spikes, gauss_filter, mode='constant')
    smooth_time_spent = scipy.ndimage.correlate(time_spent, gauss_filter, mode='constant')

    # result = spikes2 / time_spent
    smoothed_result = smooth_spikes / smooth_time_spent
    smoothed_result[time_spent < time_spent_threshold] = np.nan

    # print("smoothed_result.shape", smoothed_result.shape)
    return state.get_state().FRAME_RATE * smoothed_result

def rate_map_plot2(smoothed_result, ax=None, net="net1"):
    min_ = 0
    max_ = np.nanquantile(smoothed_result, 0.975)  # 97.5th percentile

    bin_size = state.get_state().TWO_D_PLOT_BIN_SIZE
    width, height = state.get_state().dims_per_net[net]
    x_plot_range = np.linspace(0, width // bin_size - bin_size / 2 + 1, 3)
    y_plot_range = np.linspace(0, height // bin_size - bin_size / 2 + 1, 3)

    ax.set_xticks(x_plot_range)
    ax.set_yticks(y_plot_range)

    ax.set_xticklabels((x_plot_range * bin_size + bin_size / 2).round(1))
    ax.set_yticklabels((y_plot_range * bin_size + bin_size / 2).round(1))

    if np.all(np.isnan(smoothed_result)):
        mean_fr = np.nan
        max_fr = np.nan
    else:
        mean_fr = float(np.nanmean(smoothed_result))
        max_fr = float(np.nanmax(smoothed_result))

    ax.set_title(f"Max FR: {max_fr:.2f} Hz\nMean FR: {mean_fr:.2f} Hz")  # 25Hz
    # print("smoothed_result.T.shape", smoothed_result.T.shape)
    img = ax.imshow(smoothed_result.T, cmap='jet', vmin=min_, vmax=max_)
    return mean_fr

def plot_positional_maps(axes, data2, neuron2, cols, two_d_idx):
    results = []
    for id_, i in enumerate(two_d_idx):
        # print(i, cols, two_d_idx, cols[i])
        a, b, c = cols[i].split('_')
        results.append(spike_plots.rate_map(data2, neuron2, bat_name=int(b)))

    min_ = 0
    max_ = np.nanquantile(results, 0.975)  # 97.5th percentile
    mean_frs = []
    for id_, i in enumerate(results):
        bat_id = int(cols[two_d_idx[id_]].split("_")[1])
        # print("RATE image shape", i.shape)
        mean_frs.append(spike_plots.rate_map_plot2(i, min_=min_, max_=max_, ax=axes[6 + bat_id]))
    return mean_frs

def calc_1d_feature(df, neuron, bat_name, feature_name):
    time_spent_threshold = state.get_state().ONE_DTIMESPENT_THRESHOLD 

    bins = state.get_state().ONE_D_PLOT_BIN_SIZE # change back to 12!
    # print("warning: change n_bin to 12 in spike_plots.calc_1d_feature")
    behavior = df[f"BAT_{bat_name}_F_{feature_name}"]

    min_x_value = np.floor(np.min(behavior))
    max_x_value = np.ceil(np.max(behavior))
    x_axis = np.linspace(min_x_value, max_x_value, bins)

    behavior_map = np.histogram(behavior, bins=x_axis)[0]

    not_enough_time_spent = behavior_map < time_spent_threshold
    hd_spikes_radians_map = np.histogram(behavior, bins=x_axis, weights=neuron)[0].astype('float')
    hd_spikes_radians_map[not_enough_time_spent] = np.nan

    return state.get_state().FRAME_RATE * hd_spikes_radians_map / behavior_map, x_axis

def plot_1d_feature(result, x_axis, feature_name, ax=None):
    ax.plot(x_axis[:-1], result)
    ax.set_title(f"FR: MAX={np.nanmax(result):.2f} Hz Mean={np.nanmean(result):.2f} Hz")
    ax.set_ylim(bottom=0)
    if feature_name in ["A", "HD"]:
        ax.set_xticks(np.arange(0, 360, 60))
    return np.nanmean(result)