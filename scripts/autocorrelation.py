import matplotlib.pyplot as plt
import sys
from scipy import signal
from statsmodels.tsa.stattools import pacf
import numpy as np
sys.path.append("../")

from data_manager import Loader2, DataProp1
data = Loader2()(72)
dataprop = DataProp1(data)

plt.title("Neural Data AutoCorrelation")
# plt.acorr(dataprop.data['neuron'], maxlags = 500, usevlines = True, normed = True)
# plt.show()

plt.title("Positional AutoCorrelation")
# plt.acorr(dataprop.data['BAT_0_F_X'], maxlags = 2500, usevlines = True, normed = True)
# plt.show()

plt.title("Positional AutoCorrelation")
# plt.acorr(dataprop.data['BAT_0_F_Y'], maxlags = 2500, usevlines = True, normed = True)
# plt.show()

# signal.correlate2d(dataprop.data['BAT_0_F_X'], dataprop.data['BAT_0_F_Y'])

plt.title("HD AutoCorrelation")
# plt.acorr(dataprop.data['BAT_0_F_HD'], maxlags = 2500, usevlines = True, normed = True)
# plt.show()

plt.title("partial HD AutoCorrelation")
x = dataprop.data.index
y = pacf(dataprop.data['BAT_0_F_HD'])
plt.scatter(np.arange(len(y)), y)
plt.show()