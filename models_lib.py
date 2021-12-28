import pandas as pd
import config

class Model:
	def __init__(self):
		self.covariates = None # position comprised of two covarietes: x,y
		self.features = None # position is a single 2-d feature
		self.formula = None
		self.train_test_ratio = config.Config1().train_test_ratio
		self.n_bats = config.Config1().n_bats
		self.gam_model = None

	def split_train_test(self):
		pass

	def train_model(self, data: pd.DataFrame):
		pass

	def build_formula(self):
		pass

	def build_covariates_list(self):
		pass

class AlloModel(Model):
	def __init__(self: int, covariates = None):
		super(AlloModel, self).__init__()

		if covariates is None:
			self.covariates = self.build_covariates_list()
		else:
			self.covariates = covariates

		self.formula = self.build_formula()
		self.features = []

	def __init__(self, covariates: list[str]): # to initialize sub_model
		super(AlloModel, self).__init__()
		self.covariates = covariates

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
		self.covariates.append("BAT_0_F_HD")
		for i in range(n_bats):
			self.covariates.append(f"BAT_{i}_F_X")
			self.covariates.append(f"BAT_{i}_F_Y")