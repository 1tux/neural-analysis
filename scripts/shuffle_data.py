# new elegant approach to compute shuffles
# the idea is to simply rotate the spikes and store them in a file
# then simply run the model on the rotated data.

import h5py
import numpy as np
import pandas as pd
import sys
import math
import os
sys.path.append("..")
from conf import Conf
from data_manager import neuron_id_to_day, Loader7
import data_manager
os.chdir("..")

nid = 72
day = "191220"

import glob
cells_list = glob.glob(r"C:\Users\itayy.WISMAIN\git\neural-analysis\inputs\Cells\all cells 20220209\*.mat")
cells_list_id = [int(os.path.basename(x).split("_")[0]) for x in cells_list]
cells_list_id = [3, 12, 13, 15, 16, 20, 22, 23, 24, 27, 29, 31, 46, 47, 50, 52, 53, 56, 57,
 58, 59, 61, 64, 68, 69, 72, 73, 75, 78, 80, 82, 83, 85, 87, 88, 93, 94, 95,
 96, 97, 98, 99, 101, 102, 103, 104, 133, 134, 144, 145, 146, 147, 148, 149,
 150, 151, 152, 153, 159, 161, 163, 164, 167, 168, 169, 170, 171, 173, 174,
 191,  197, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 215,
 218, 220, 225, 226, 227, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388,
 389, 390, 391, 392, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427,
 428]
cells_list_id = [387, 388, 389, 390, 391, 392, 417, 418, 419, 420, 421, 422,
 423, 424, 425, 426, 427, 428]

cells_list_id = [381]

print(f"RUNNING {len(cells_list_id)} CELLS")

for nid in cells_list_id:
	day = neuron_id_to_day(nid)
	if nid < 228:
		bat = "2305"
	elif nid > 377:
		bat = "7757"
	else:
		exit("Add Bat ID")
	print(nid)

	neuron_path = f"inputs/Cells/{nid}_b{bat}_{day}_cell_analysis.mat"
	d = h5py.File(neuron_path, "r")
	spikes = np.array(d['cell_analysis']['spikes_per_frame']).T[0]

	# open corresponding behavioral file
	data = Loader7()(nid)

	dataprop = data_manager.DataProp1(data)

	start = Conf().SHUFFLES_MIN_GAP
	assert len(dataprop.data['neuron']) > 2 * Conf().SHUFFLES_MIN_GAP
	end = len(dataprop.data['neuron']) - Conf().SHUFFLES_MIN_GAP
	shift_size = math.ceil((end - start) / Conf().SHUFFLES_JMPS)

	POWERS_OF_TWO = [2**i for i in range(20)]
	shuffles_list = [0] + list(range(start, end, shift_size))
	# print(len(shuffles_list))
	for i, shift in enumerate(shuffles_list):
		if i in POWERS_OF_TWO: print(i)
		neuron_path2 = neuron_path.replace(f"/{nid}_b", f"/{nid}_{shift}_b").replace("Cells", "shuffles").replace(".mat", ".csv")
		if True or not os.path.exists(neuron_path2):
			#spikes2 = np.roll(spikes, shift)
			#spikes2_ = pd.Series(spikes2)
			#spikes2_.to_csv(neuron_path2)

			spikes2 = np.roll(dataprop.data['neuron'], shift)

			zeros = pd.Series([0] * len(spikes))
			zeros.loc[dataprop.no_nans_indices] = spikes2
			spikes2_ = zeros
			# spikes2_ = pd.Series(spikes2, dataprop.no_nans_indices)
			# print(np.sum(dataprop.data['neuron']), np.sum(spikes2))
			assert np.sum(dataprop.data['neuron']) == np.sum(spikes2)
			spikes2_.to_csv(neuron_path2)
