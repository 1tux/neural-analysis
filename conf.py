""" Refelects the execution state.
    Stores configurations that might be override with arguments and runtime global variables.
"""

# Conf() is implemented using Singleton
class Conf(object):
   def __new__(cls):
       if not hasattr(cls, 'instance'):
           cls.instance = super(Conf, cls).__new__(cls)

       return cls.instance

# CONSTS
Conf().TRAIN_TEST_RATIO = 0.5
Conf().DIMS_PER_NET = {
    "net1" : (100, 50),
    "net3" : (50, 50)
}

Conf().FRAME_RATE = 25

Conf().ONE_D_PLOT_BIN_SIZE = 12
Conf().ONE_D_TIME_SPENT_THRESHOLD = 1 #13 # 0.520 second
Conf().TWO_D_PLOT_BIN_SIZE = 3
Conf().TWO_D_TIME_SPENT_THRESHOLD = 1 # 5
Conf().GAUSSIAN_FILTER_SIGMA = 2.5
Conf().GAUSSIAN_FILTER_SIZE = 5 * (round(Conf().GAUSSIAN_FILTER_SIGMA) + 1)  # 3cm
Conf().TWO_D_PRECENTILE_CUTOFF = 0.975
Conf().TIME_BASED_GROUP_SPLIT = Conf().FRAME_RATE * 5 # 5 seconds seems to be the correct time window to avoid autocorrelation

Conf().SHUFFLES_MIN_GAP = 10 * 60 * Conf().FRAME_RATE  # 10 minutes
Conf().SHUFFLES_JMPS = 100

# configuration related to matlab file parsing. sizes of nets.
Conf().NET1_MIN_X = 150
Conf().NET1_MAX_X = 250
Conf().NET1_MIN_Y = 10
Conf().NET1_MAX_Y = 60
Conf().NET3_MIN_X = 25
Conf().NET3_MAX_X = 75
Conf().NET3_MIN_Y = 170
Conf().NET3_MAX_Y = 220

# PATHS
Conf().INPUT_FOLDER = "inputs/"
Conf().IMGS_PATH = "imgs/"
Conf().CACHE_FOLDER = "cache/"

# DEFAULT ARGS
Conf().USE_CACHE = True
Conf().TO_PLOT = True
Conf().SHUFFLES = 0 # default number of shuffles
Conf().RUN_SHAPLEY = False
Conf().FEATURE_SELECTION = True