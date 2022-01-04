
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

    def process(self):
        self.scale_maps()

    def scale_maps(self):
        print("Scaling models maps...")
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
        model_pos_axis = g.model_pos_axis
        hd_axis = g.hd_axis
        angle_axis = g.angle_axis
        dis_axis = g.dis_axis
        fr_axis = g.fr_axis
        # firing_rate_map = rate_maps.FiringRate(self.dataprop)

        idx_pos = 0
        idx_pos2 = 0
        idx_dis = 0
        idx_angle = 0

        for m in maps.values():
            
            if m.feature_type == "fr":
                m.plot(fr_axis)

            elif m.feature.type_ == features_lib.FeatureType.POS and isinstance(m, rate_maps.RateMap2D):
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