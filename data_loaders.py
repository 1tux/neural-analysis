import pandas as pd

class DataLoader:
	pass

class Loader1(DataLoader):
	def __init__(self):
		pass

	def __call__(self, nid: int) -> pd.DataFrame:
		df = pd.read_csv(r"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\b2305_d191220_simplified_behaviour.csv").drop(columns=['Unnamed: 0'])
		df['neuron'] = pd.read_csv(r"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\72_b2305_d191220.csv")['0']
		return df