import plot_lib
from conf import Conf
import rate_maps
import matplotlib.pyplot as plt
import numpy as np

class Results:
    def plot(self):
        pass

    def store(self):
        pass

class Results1(Results):
    def __init__(self):
        self.g = None

    def process(self):
        self.scale_maps()

    def scale_maps(self):
        print("Scaling models maps...")
        print(self.rate_maps.keys())
        print(self.models_maps.keys())
        for feature_name in self.models_maps:
            rate_map = self.rate_maps[feature_name]
            model_map = self.models_maps[feature_name]
            self.models_maps[feature_name].map_ = self.scale_map(model_map, rate_map)

    def scale_map(self, model_map, rate_map):
        rate_map_mean = np.nanmean(rate_map.map_)
        model_map_mean = np.nanmean(model_map.map_)
        return model_map.map_ / model_map_mean * rate_map_mean # forces the means to be equal

    def plot(self):
        self.g = plot_lib.PlotGrid(self.dataprop.n_bats)
        self.plot1(self.rate_maps)
        self.plot1(self.models_maps)
        plt.show()

    def plot1(self, maps):
        g = self.g
        pos_axis = g.pos_axis
        hd_axis = g.hd_axis
        angle_axis = g.angle_axis
        dis_axis = g.dis_axis

        firing_rate_map = rate_maps.FiringRate(self.dataprop)
        firing_rate_map.plot(g.fr_axis)

        idx_pos = 0
        idx_dis = 0
        idx_angle = 0

        for m in maps.values():
            if m.feature_type == "POS":
                m.plot(pos_axis[idx_pos])
                idx_pos += 1

            elif m.feature_type == "HD":
                m.plot(hd_axis)

            elif m.feature_type == "D":
                m.plot(dis_axis[idx_dis])
                idx_dis += 1

            elif m.feature_type == "A":
                m.plot(angle_axis[idx_angle])
                idx_angle += 1
            else:
                print(m.feature_type)