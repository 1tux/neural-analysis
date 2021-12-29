""" Refelects the execution state.
    Stores configurations that might be override with arguments and runtime global variables.
"""

# State is implemented using Singleton
class State(object):
   def __new__(cls):
       if not hasattr(cls, 'instance'):
           cls.instance = super(State, cls).__new__(cls)

       return cls.instance

State().train_test_ratio = 0.5
State().dims_per_net = {
    "net1" : (100, 50),
    "net3" : (50, 50)
}
State().ONE_D_PLOT_BIN_SIZE = 12
State().ONE_DTIMESPENT_THRESHOLD = 13  # 0.520 second
State().TWO_D_PLOT_BIN_SIZE = 3
State().TWO_D_TIME_SPENT_THRESHOLD = 5
State().GAUSSIAN_FILTER_SIGMA = 2.5
State().GAUSSIAN_FILTER_SIZE = 5 * (round(State().GAUSSIAN_FILTER_SIGMA) + 1)  # 3cm
State().FRAME_RATE = 25

State().n_bats = None # to be deduced in running time
State().no_nans_indices = None # indices before removing nans