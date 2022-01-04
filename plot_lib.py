import matplotlib.pyplot as plt

from conf import Conf
import rate_maps
class PlotGrid:
    def __init__(self, n_bats):
        fig = plt.figure()
        grid = plt.GridSpec(5, n_bats, wspace=0.4, hspace=0.3)

        self.fr_axis = fig.add_subplot(grid[0, :n_bats])
        self.pos_axis = [fig.add_subplot(grid[1, i]) for i in range(n_bats)]
        self.model_pos_axis = [fig.add_subplot(grid[2, i]) for i in range(n_bats)]
        self.hd_axis = fig.add_subplot(grid[3, 0])
        self.angle_axis = [fig.add_subplot(grid[3, i]) for i in range(1, n_bats)]
        self.dis_axis = [fig.add_subplot(grid[4, i]) for i in range(1, n_bats)]