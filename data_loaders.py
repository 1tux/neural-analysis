import pandas as pd

class DataLoader:
	pass

class Loader1(DataLoader):
	def __init__(self):
		pass

	def __call__(self, nid: int) -> pd.DataFrame:
		return pd.read_csv(r"C:\Users\root34\Documents\university\MSc\Bat Lab\git\neural-analysis\inputs\b2305_d191220_simplified_behaviour.csv")
		return None