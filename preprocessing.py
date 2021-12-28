import pandas as pd
import numpy as np

class Preprocess:
	def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
		return data

class Preprocess1(Preprocess):
	def __init__(self):
		pass

	def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
		return add_pairwise_features(remove_nans(data))

def remove_nans(data):
	return data.dropna()

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
	pass