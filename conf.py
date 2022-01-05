""" Refelects the execution state.
    Stores configurations that might be override with arguments and runtime global variables.
"""

# Conf() is implemented using Singleton
class Conf(object):
   def __new__(cls):
       if not hasattr(cls, 'instance'):
           cls.instance = super(Conf, cls).__new__(cls)

       return cls.instance

Conf().TRAIN_TEST_RATIO = 0.5
Conf().DIMS_PER_NET = {
    "net1" : (100, 50),
    "net3" : (50, 50)
}
Conf().ONE_D_PLOT_BIN_SIZE = 12
Conf().ONE_D_TIME_SPENT_THRESHOLD = 13  # 0.520 second
Conf().TWO_D_PLOT_BIN_SIZE = 3
Conf().TWO_D_TIME_SPENT_THRESHOLD = 5
Conf().GAUSSIAN_FILTER_SIGMA = 2.5
Conf().GAUSSIAN_FILTER_SIZE = 5 * (round(Conf().GAUSSIAN_FILTER_SIGMA) + 1)  # 3cm
Conf().FRAME_RATE = 25
Conf().TWO_D_PRECENTILE_CUTOFF = 0.975
Conf().CACHE_FOLDER = "cache/"