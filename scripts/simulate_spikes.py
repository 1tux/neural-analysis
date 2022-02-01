import argparse
import sys
import spike_gen
import os
import os.path
import numpy as np
import pandas as pd
import itertools
import data_manager
from conf import Conf

def main(day, output_dir):
    df = pd.read_csv(os.path.join("..", Conf().INPUT_FOLDER, fr"b2305_{day}_simplified_behaviour.csv")).drop(columns=['Unnamed: 0'])

    output_path = os.path.join("..", Conf().INPUT_FOLDER, "simulated", day)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # output_path = os.path.join(output_dir, day)

    # neuron1 - Place cell
    simulated_spike = spike_gen.gaussian_place_cell(df, 80, 35, 22)
    simulated_spike.to_csv(os.path.join(output_path, "A_place_cell.csv"))

    # neuron2 - HD + place cell
    simulated_spike = spike_gen.gaussian_place_cell(df, 80, 35, 22)
    simulated_spike &= spike_gen.gaussian_head_direction_cell(df, 80, 15)
    simulated_spike.to_csv(os.path.join(output_path, "HD_place_cell.csv"))

    # neuron2 - HD + place cell + pos1
    simulated_spike = spike_gen.gaussian_place_cell(df, 80, 35, 22)
    simulated_spike &= spike_gen.gaussian_head_direction_cell(df, 80, 15)
    simulated_spike = spike_gen.gaussian_place_cell(df, 80, 35, 22, bat_name=1)
    simulated_spike.to_csv(os.path.join(output_path, "HD_place_cell_pos1.csv"))


    # neuron3 - place cell + distance(1)
    simulated_spike = spike_gen.gaussian_place_cell(df, 80, 35, 22)
    simulated_spike &= spike_gen.gaussian_distance_cell(df, 25, 1, 5)
    simulated_spike.to_csv(os.path.join(output_path, "place_distance1_cell.csv"))

    # neuron4 - place cell + distance(2), 3, 4
    for i in range(2, 5):
        simulated_spike = spike_gen.gaussian_place_cell(df, 80, 35, 22)
        simulated_spike &= spike_gen.gaussian_distance_cell(df, 25, i, 5)
        simulated_spike.to_csv(os.path.join(output_path, f"place_distance{i}_cell.csv"))

    # neuron5 - random neuron
    simulated_spike = pd.Series(np.random.poisson(lam=0.5, size=len(simulated_spike)))
    simulated_spike.to_csv(os.path.join(output_path, "randomly_firing_cell.csv"))

    # neuron6 - pairwise distances
    for p in itertools.combinations([1,2,3,4], 2): # all pairs
        simulated_spike = spike_gen.pairwise_distance_cell(df, p[0], p[1], 25, 5)
        simulated_spike.to_csv(os.path.join(output_path, f"pairwise_distance{p[0]},{p[1]}_cell.csv"))

    # neuron7 - distance and angles
    for i in range(1, 5):
        simulated_spike = spike_gen.gaussian_angle_cell(df, 80, i, 22)
        simulated_spike &= spike_gen.gaussian_distance_cell(df, 25, i, 5)
        simulated_spike.to_csv(os.path.join(output_path, f"ego_cell{i}.csv")) 

    # neuron8 - distance and angles for two bats
    simulated_spike = spike_gen.gaussian_angle_cell(df, 80, 1, 22)
    simulated_spike &= spike_gen.gaussian_distance_cell(df, 25, 1, 5)
    simulated_spike = spike_gen.gaussian_angle_cell(df, 80, 2, 22)
    simulated_spike &= spike_gen.gaussian_distance_cell(df, 25, 2, 5)
    simulated_spike.to_csv(os.path.join(output_path, f"ego_cell_1_2.csv"))    

if __name__ == "__main__":
    days = [f"d19122{i}" for i in range(7)] + ["d191229", "d191231", "d200101"]
    for day in days:
        print(f"simulating spikes for day: {day}")
        main(day, r"C:\Users\itayy.WISMAIN\git\neural-analysis\inputs\simulated")