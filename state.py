class get_state(object):
   def __new__(cls):

       if not hasattr(cls, 'instance'):
           cls.instance = super(get_state, cls).__new__(cls)

       return cls.instance

get_state().train_test_ratio = 0.5
get_state().dims_per_net = {
    "net1" : (100, 50),
    "net3" : (50, 50)
}
get_state().ONE_D_PLOT_BIN_SIZE = 12
get_state().ONE_DTIMESPENT_THRESHOLD = 13  # 0.520 second
get_state().TWO_D_PLOT_BIN_SIZE = 3
get_state().TWO_D_TIME_SPENT_THRESHOLD = 5
get_state().GAUSSIAN_FILTER_SIGMA = 2.5
get_state().GAUSSIAN_FILTER_SIZE = 5 * (round(get_state().GAUSSIAN_FILTER_SIGMA) + 1)  # 3cm
get_state().FRAME_RATE = 25