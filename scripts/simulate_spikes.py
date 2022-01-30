import argparse
import sys
import spike_gen
import analysis_lib
import config
import os
import os.path
import numpy as np
import pandas as pd
import itertools

def handle_args(args):
    parser = argparse.ArgumentParser(description='Simulate spikes & Analyze.')
    # parser.add_argument('bpath', metavar='behavioral_path', type=str, nargs='?', help='Path for behavioral data (mat/csv)', default=r"C:\Users\itayy\Documents\saikat - Bat Lab\data\behavioural_data\parsed\b2305_d191223_simplified_behaviour.csv")
    parser.add_argument('bpath', metavar='behavioral_path', type=str, nargs='?', help='Path for behavioral data (mat/csv)', default=r"C:\Users\itayy\Documents\saikat - Bat Lab\data\behavioural_data\parsed\b2305_d191220_simplified_behaviour.csv")
    parser.add_argument('-n', metavar='net', type=int, help='which net, could be 1 or 3', default=1)
    parser.add_argument('-X', metavar='eXclude', type=str, nargs='*', default=[])
    parser.add_argument('cpath', metavar='config_path', type=str, nargs='?', help='Path for configuration file (json)', default='config.json')
    args = parser.parse_args()

    config.Config.from_file(args.cpath)
    behavioral_data_path = args.bpath
    exclude = args.X

    try:
        net = {1: "NET1", 3: "NET3"}[args.n]
    except:
        raise Exception("Wrong Net! should have been either 1 or 3 %s" % str(args.n))

    return behavioral_data_path, net, exclude

    bat_name, day, _, _ = Path(behavioral_data_path).stem.split('_')

def main(behavioral_data_path, net, exclude, folder_path = None):
    dataset = analysis_lib.behavioral_data_to_dataframe(behavioral_data_path, net, exclude)
    if folder_path is None:
        folder_path = r"C:\Users\itayy\Documents\Bat-Lab\data\simulated_neural_data"

    # neuron1 - place cell
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22)
    simulated_spike &= spike_gen.gaussian_head_direction_cell(dataset, 80, 15)
    neuron1_path = os.path.join(folder_path, "HD_place_cell.csv")
    simulated_spike.to_csv(neuron1_path)

    # neuron2 - OR place cell Pair23
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 3)
    simulated_spike |= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 2)
    neuron2_path = os.path.join(folder_path, "OR_place_cell.csv")
    simulated_spike.to_csv(neuron2_path)
    
    # neuron3 - AND place cell Pair 23
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 3)
    simulated_spike &= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 2)
    neuron3_path = os.path.join(folder_path, "AND_place_cell.csv")
    simulated_spike.to_csv(neuron3_path)

    # neuron4 - Place cell
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22)
    neuron4_path = os.path.join(folder_path, "A_place_cell.csv")
    simulated_spike.to_csv(neuron4_path)

    # neuron5 - Place cell different location
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 50, 35, 22)
    neuron5_path = os.path.join(folder_path, "A_place_cell2.csv")
    simulated_spike.to_csv(neuron5_path)

    # neuron6 - Any Other place cell different
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 1)
    simulated_spike |= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 2)
    simulated_spike |= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 3)
    simulated_spike |= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 4)
    neuron6_path =  os.path.join(folder_path, "Anyother_place_cell.csv")
    simulated_spike.to_csv(neuron6_path)

    # neuron7 - Any place cell different
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 1)
    simulated_spike |= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 2)
    simulated_spike |= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 3)
    simulated_spike |= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 4)
    simulated_spike |= spike_gen.gaussian_place_cell(dataset, 80, 35, 22, 0)
    neuron7_path = os.path.join(folder_path, "Any_place_cell.csv")
    simulated_spike.to_csv(neuron7_path)

    # neuron8 - Distance and Angle
    simulated_spike = spike_gen.gaussian_angle_cell(dataset, 35, 1, 25)
    simulated_spike &= spike_gen.gaussian_distance_cell(dataset, 25, 1, 5)
    neuron8_path = os.path.join(folder_path, "an_ego_cell.csv")
    simulated_spike.to_csv(neuron8_path)


def main2(behavioral_data_path, net, exclude, folder_path = None):
    dataset = analysis_lib.behavioral_data_to_dataframe(behavioral_data_path, net, exclude)
    if folder_path is None:
        folder_path = r"C:\Users\itayy\Documents\Bat-Lab\data\simulated_neural_data"

    # neuron1 - Place cell
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22)
    simulated_spike.to_csv(os.path.join(folder_path, "A_place_cell.csv"))

    # neuron2 - HD + place cell
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22)
    simulated_spike &= spike_gen.gaussian_head_direction_cell(dataset, 80, 15)
    simulated_spike.to_csv(os.path.join(folder_path, "HD_place_cell.csv"))

    # neuron3 - place cell + distance(1)
    simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22)
    simulated_spike &= spike_gen.gaussian_distance_cell(dataset, 25, 1, 5)
    simulated_spike.to_csv(os.path.join(folder_path, "place_distance1_cell.csv"))

    # neuron4 - place cell + distance(2), 3, 4
    for i in range(2, 5):
        simulated_spike = spike_gen.gaussian_place_cell(dataset, 80, 35, 22)
        simulated_spike &= spike_gen.gaussian_distance_cell(dataset, 25, i, 5)
        simulated_spike.to_csv(os.path.join(folder_path, f"place_distance{i}_cell.csv"))

    # neuron5 - random neuron
    simulated_spike = pd.Series(np.random.poisson(lam=0.5, size=len(simulated_spike)))
    simulated_spike.to_csv(os.path.join(folder_path, "randomly_firing_cell.csv"))

    # neuron6 - pairwise distances
    for p in itertools.combinations([1,2,3,4], 2): # all pairs
        simulated_spike = spike_gen.pairwise_distance_cell(dataset, p[0], p[1], 25, 5)
        simulated_spike.to_csv(os.path.join(folder_path, f"pairwise_distance{p[0]},{p[1]}_cell.csv"))

if __name__ == "__main__":
    behavioral_data_path, net, exclude = handle_args(sys.argv)
    days = [f"19122{i}" for i in range(7)] + ["191229", "191231", "200101"]
    for d in days:
        behavioral_data_path = fr"C:\Users\itayy\Documents\saikat - Bat Lab\data\behavioural_data\parsed\b2305_d{d}_simplified_behaviour.csv"
        behavioral_data_path = fr"C:\Users\itayy\Documents\Bat-Lab\data\behavioral_data\parsed\b2305_d{d}_simplified_behaviour.csv"
        output_dir = fr"C:\Users\itayy\Documents\Bat-Lab\data\simulated_neural_data - d{d}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        main2(behavioral_data_path, net, exclude, output_dir)