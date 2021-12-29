import matplotlib.pyplot as plt

from state import State
import preprocessing

class Map:
    def __init__(self, data):
        self.data = data
        self.calculate_map()

    def calculate_map(self):
        pass

    def plot(self):
        pass

class FiringRate(Map):
    def calculate_map(self):
        self.x = x = State().no_nans_indices
        fr = preprocessing.spikes_to_firing_rate(self.data['neuron'])
        self.y = State().FRAME_RATE * fr

    def plot(self, ax):
        ax.plot(self.x, self.y, '.', markersize=1, alpha=0.5, label='test-firing-rates')

class PlotGrid:
    def __init__(self):
        n_bats = State().n_bats

        fig = plt.figure()
        grid = plt.GridSpec(4, n_bats, wspace=0.4, hspace=0.3)

        self.fr_axis = fig.add_subplot(grid[0, :n_bats])
        self.pos_axis = [fig.add_subplot(grid[1, i]) for i in range(n_bats)]
        self.hd_axis = fig.add_subplot(grid[2, 0])
        self.angle_axis = [fig.add_subplot(grid[2, i]) for i in range(1, n_bats)]
        self.dis_axis = [fig.add_subplot(grid[3, i]) for i in range(1, n_bats)]

    def plot_firing_rate(self, map: FiringRate):
        self.fr_axis.plot(map.x, map.y, '.', markersize=1, alpha=0.5, label='test-firing-rates')

    def plot(self):
        self.plot_firing_rate()



def plot_data_based_maps(maps):
    pass