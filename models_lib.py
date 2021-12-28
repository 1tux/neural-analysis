from __future__ import annotations
import pandas as pd
import state

class Model:
	def __init__(self, covariates = None, use_cache=True):
		self.covariates = None # position comprised of two covarietes: x,y

		if covariates is None:
			self.covariates = self.build_covariates_list()
		else:
			self.covariates = covariates

		self.features = None # position is a single 2-d feature
		self.formula = None
		self.train_test_ratio = state.get_state().train_test_ratio
		self.n_bats = state.get_state().n_bats
		self.gam_model = None
		self.score = None
		self.use_cache = use_cache

	def split_train_test(self):
		pass

	def train_model(self, data: pd.DataFrame):
		pass

	def build_formula(self):
		pass

	def build_covariates_list(self):
		pass

	def evaulate(self):
		self.score = 10

	# retrain sub-models over all the covariets
	# estimate shapley values
	def shapley(self):
		pass

	def generate_maps(self):
		pass

	def run_shuffles(self):
		pass

class AlloModel(Model):
	def build_covariates_list(self):
		"""
		example in case there are 5 bats
		[
				'BAT_0_F_X', 'BAT_0_F_Y', 'BAT_0_F_HD',
                'BAT_1_F_X', 'BAT_1_F_Y',
                'BAT_2_F_X', 'BAT_2_F_Y',
                'BAT_3_F_X', 'BAT_3_F_Y',
                'BAT_4_F_X', 'BAT_4_F_Y'
		]"""
		self.covariates = ["BAT_0_F_HD"]
		for i in range(state.get_state().n_bats):
			self.covariates.append(f"BAT_{i}_F_X")
			self.covariates.append(f"BAT_{i}_F_Y")

class EgoModel(Model):
	def build_covariates_list(self):
		"""
		example in case there are 5 bats
		[
				'BAT_0_F_X', 'BAT_0_F_Y', 'BAT_0_F_HD',
                'BAT_1_F_X', 'BAT_1_F_Y',
                'BAT_2_F_X', 'BAT_2_F_Y',
                'BAT_3_F_X', 'BAT_3_F_Y',
                'BAT_4_F_X', 'BAT_4_F_Y'
		]"""
		self.covariates = []
		for i in range(state.get_state().n_bats):
			self.covariates.append(f"BAT_{i}_F_A")
			self.covariates.append(f"BAT_{i}_F_D")

def get_best_model(sub_models: list[models_lib.Model], sub_data: list[pd.DataFrame]) -> models_lib.Model:
	scores = []
	for model, data in zip(sub_models, sub_data):
		model.train_model(data)
		model.evaulate()
		scores.append((model.score, model))

	# best model is the one with the lowest score
	best_model = sorted(scores, key=lambda i:i[0])[0][1]
	return best_model