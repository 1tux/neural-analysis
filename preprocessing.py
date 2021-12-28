import pandas as pd

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

def add_pairwise_features(data):
	return data

def get_number_of_bats(data: pd.DataFrame) -> int:
	return 5

def spikes_to_firing_rate(spikes):
	pass

def maps(data):
	pass