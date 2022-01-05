
import matplotlib.pyplot as plt
import numpy as np


from conf import Conf
import features_lib
import rate_maps
import model_maps
import plot_lib


class Results:
    def plot(self):
        pass

    def store(self):
        pass

class Results1(Results):
    def __init__(self):
        self.g = None

    def plot(self):
        self.g = plot_lib.PlotGrid(self.dataprop.n_bats, self.best_model)
        my_rate_maps = {k: self.data_maps[k] for k in self.models_maps}
        self.plot_maps(my_rate_maps)
        self.fr_map.plot(self.g.fr_axis)
        self.plot_maps(self.models_maps)
        self.model_fr_map.plot(self.g.fr_axis)

        plt.show()

    def plot_maps(self, maps):
        g = self.g
        pos_axis = g.pos_axis
        model_pos_axis = g.model_pos_axis
        hd_axis = g.hd_axis
        angle_axis = g.angle_axis
        dis_axis = g.dis_axis

        idx_pos = 0
        idx_pos2 = 0
        idx_dis = 0
        idx_angle = 0

        for m in maps.values():
            if m.feature.type_ == features_lib.FeatureType.POS and isinstance(m, rate_maps.RateMap2D):
                m.plot(pos_axis[idx_pos])
                idx_pos += 1

            elif m.feature.type_ == features_lib.FeatureType.POS and isinstance(m, model_maps.ModelMap2D):
                m.plot(model_pos_axis[idx_pos2])
                idx_pos2 += 1

            elif m.feature.type_ == features_lib.FeatureType.HD:
                m.plot(hd_axis)

            elif m.feature.type_ == features_lib.FeatureType.D:
                m.plot(dis_axis[idx_dis])
                idx_dis += 1

            elif m.feature.type_ == features_lib.FeatureType.A:
                m.plot(angle_axis[idx_angle])
                idx_angle += 1

            else:
                print(m.feature_type)