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
from data_manager import neuron_id_to_day, Loader9
import data_manager
import glob
import tqdm

os.chdir("..")

ALL_CELLS = glob.glob("inputs/cells/*.csv")
cells_list_id = set(list(map(int, open("scripts/cells_to_shuffle.txt","rb").readlines()))) # CELLS to run shuffle on


def get_spikes_and_behavior(cell_path, batname, day):
	# loads raw spikes and behavior.
	# however we need the data after NANs removal!
	spikes = pd.read_csv(cell_path)['0']
	behavior = pd.read_csv(f"../inputs/{batname}_{day}.csv")
	return spikes, behavior

def get_spikes(nid, cell_path):
	spikes_with_gaps = pd.read_csv(cell_path)['0']

	data = Loader9()(nid)
	dataprop = data_manager.DataProp1(data) # do NANs removal for us
	spikes_without_gaps = dataprop.data['neuron']
	# we don't really need the behavior after NAN removal
	# behavior = dataprop.data.drop(columns=['neuron'])
	return spikes_with_gaps, spikes_without_gaps, dataprop.no_nans_indices

def compute_shuffles(spikes_with_gaps, spikes_without_gaps, no_nans_indices):
	start = Conf().SHUFFLES_MIN_GAP
	assert len(spikes_without_gaps) > 2 * Conf().SHUFFLES_MIN_GAP
	end = len(spikes_without_gaps) - Conf().SHUFFLES_MIN_GAP
	shift_size = math.ceil((end - start) / Conf().SHUFFLES_JMPS)

	shuffles_list = [0] + list(range(start, end, shift_size))
	result = {}
	for shift in shuffles_list:
		shuffled_spikes_without_gaps = np.roll(spikes_without_gaps, shift)

		tmp_with_gaps = pd.Series([0] * len(spikes_with_gaps))
		tmp_with_gaps.loc[no_nans_indices] = shuffled_spikes_without_gaps
		shuffled_spikes_with_gaps = tmp_with_gaps
		result[shift] = shuffled_spikes_with_gaps
	return result


def store_shuffles(shuffles, output_dir):
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	for shift, shuffle in tqdm.tqdm(shuffles.items()):
		pd.Series(shuffle).to_csv(os.path.join(output_dir, f"{shift}.csv"), index=False)

ctr = 0
for filename in ALL_CELLS:
	nid, batname, day = os.path.basename(filename).split('_')
	day = day[:-4] # not including .csv
	nid = int(nid)

	if nid in cells_list_id:
		print(f"Shuffling neuron {nid} ({ctr+1}/{len(cells_list_id)})")
		# spikes, behavior = get_spikes_and_behavior(filename, batname, day)
		# spikes_with_gaps, spikes_without_gaps, no_nans_indices = get_spikes(nid, filename)
		shuffles = compute_shuffles(*get_spikes(nid, filename))
		store_shuffles(shuffles, os.path.join("inputs", "shuffles", f"{nid}"))
		ctr += 1

if ctr != len(cells_list_id):
	print(f"Did not run all cells! ({ctr}/{len(cells_list_id)})")
	exit(-1)

exit(0)


print(f"RUNNING {len(cells_list_id)} CELLS")


os.chdir("..") # for Loader

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
