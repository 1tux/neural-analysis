import matplotlib.pyplot as plt

from conf import Conf
import rate_maps
class PlotGrid:
    def __init__(self, n_bats):

        fig = plt.figure()
        grid = plt.GridSpec(4, n_bats, wspace=0.4, hspace=0.3)

        self.fr_axis = fig.add_subplot(grid[0, :n_bats])
        self.pos_axis = [fig.add_subplot(grid[1, i]) for i in range(n_bats)]
        self.hd_axis = fig.add_subplot(grid[2, 0])
        self.angle_axis = [fig.add_subplot(grid[2, i]) for i in range(1, n_bats)]
        self.dis_axis = [fig.add_subplot(grid[3, i]) for i in range(1, n_bats)]

    def plot_firing_rate(self, map_: rate_maps.FiringRate):
        self.fr_axis.plot(map_.x, map_.y, '.', markersize=1, alpha=0.5, label='test-firing-rates')

    def plot(self):
        self.plot_firing_rate()

def plot_data_based_maps(maps):
    pass