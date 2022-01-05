import matplotlib.pyplot as plt

from conf import Conf
import models
class PlotGrid:
    def __init__(self, n_bats, model_type: models.Model):
        fig = plt.figure()
        self.fr_axis = self.pos_axis = self.model_pos_axis = self.hd_axis = self.angle_axis = self.dis_axis = None
        if isinstance(model_type,  models.AlloModel):
            grid = plt.GridSpec(4, n_bats, wspace=0.4, hspace=0.3)

            self.fr_axis = fig.add_subplot(grid[0, :n_bats])
            self.pos_axis = [fig.add_subplot(grid[1, i]) for i in range(n_bats)]
            self.model_pos_axis = [fig.add_subplot(grid[2, i]) for i in range(n_bats)]
            self.hd_axis = fig.add_subplot(grid[3, 0])

        elif isinstance(model_type, models.EgoModel):
            grid = plt.GridSpec(3, n_bats-1, wspace=0.4, hspace=0.3)

            self.fr_axis = fig.add_subplot(grid[0, :n_bats])
            self.angle_axis = [fig.add_subplot(grid[1, i]) for i in range(n_bats-1)]
            self.dis_axis = [fig.add_subplot(grid[2, i]) for i in range(n_bats-1)]
        

