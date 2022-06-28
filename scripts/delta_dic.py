import glob
import os

to_filenames = lambda l: [os.path.basename(x).split(".")[0] for x in l]

# PATH FOR GAM EXECUTED CELLS
allo_cells = set(to_filenames(glob.glob(r"C:\tmp\shuffles_dist_allo\*")))
ego_cells = set(to_filenames(glob.glob(r"C:\tmp\shuffles_dist_ego\*")))

shared_cells = sorted(map(int, allo_cells & ego_cells))
#print(shared_cells)
#print(len(shared_cells))
#print(len(allo_cells))
#print(len(ego_cells))

sig_cells = [57, 58, 61, 64, 68, 72, 73, 78, 133, 144, 145, 148, 149, 150, 151, 153, 159, 170, 171, 210, 215, 390, 419, 420]
sig_cells = [57, 58, 61, 64, 68, 72, 73, 78, 144, 145, 150, 153, 159, 170, 171, 210, 215, 419, 420]
#sig_cells = [57, 58, 61, 64, 72, 73, 144, 147, 153, 170, 171, 215, 418, 420]

for nid in sig_cells:
	allo = open(f"../stats_allo/{nid}_AlloModel_11111111111_0_CV.txt").readlines()
	ego = open(f"../stats_ego/{nid}_EgoModel_11111111_0_CV.txt").readlines()
	for l in allo:
		if l.startswith('DIC'):
			allo_dic = float(l.split("DIC: ")[1])
			#print(nid, l, end=" ")
	for l in ego:
		if l.startswith('DIC'):
			ego_dic = float(l.split("DIC: ")[1])
			#print(nid, l, end=" ")
	print(nid, allo_dic - ego_dic)
