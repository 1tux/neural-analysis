import pandas as pd
from dataclasses import dataclass
from enum import Enum
import typing

def get_feature_name(bat_name, feature_name):
    return f"BAT_{bat_name}_F_{feature_name}"

def extract_bats_names(df: pd.DataFrame) -> typing.List[str]:
    bats_names = pd.Series(df.columns.str.extract("BAT_(\d)_*", expand=False).unique()).dropna().values
    return bats_names

def get_n_bats(covariates_list):
    return len(set([i.split('_')[1] for i in covariates_list]))

get_label_name = lambda: "neuron"

class FeatureType(Enum):
    HD = 1
    A = 2
    D = 3
    POS = 4

    def dim(self):
        one_d = [self.HD, self.A, self.D]
        two_d = [self.POS]
        if self in one_d:
            return 1

        if self in two_d:
            return 2

@dataclass()
class Feature:
    name: str
    covariates: tuple
    covariates_indices: tuple
    type_: FeatureType = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def dim(self):
        return self.type_.dim()


def covariates_to_features(covariates):
    features = []
    for covariate_idx, covariate in enumerate(covariates):
        suffix = covariate.split("_")[-1]
        if suffix == "X":
            y_idx = covariates.index(covariate[:-1]+"Y")
            y_covariate_name = covariate.replace("_X", "_Y")
            pos_feature_name = (covariate.replace("_X", "_POS"))
            f = Feature(pos_feature_name, (covariate, y_covariate_name), (covariate_idx, y_idx))
        elif suffix == "Y":
            continue
        else:
            f = Feature(covariate, (covariate, ), (covariate_idx, ))
        f.type_ = FeatureType[f.name.split("_")[-1]]
        features.append(f)
    return features

def features_to_covariates(features):
    covarites = list(map(lambda f: list(f.covariates), features))
    flat_covarites = [item for sublist in covarites for item in sublist]
    return sorted(flat_covarites)
