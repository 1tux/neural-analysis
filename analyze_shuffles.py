import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats_lib
import math

nid = 72

neurons = [72]
neurons = [3, 12, 13, 15, 16, 20, 22, 23, 24, 27, 29, 31, 46, 47, 50, 52, 53, 56, 57,
 58, 59, 61, 64, 68, 69, 72, 73, 75, 78, 80, 82, 83, 85, 87, 88, 93, 94, 95,
 96, 97, 98, 99, 101, 102, 103, 104, 133, 134, 144, 145, 146, 147, 148, 149,
 150, 151, 152, 153, 159, 161, 163, 164, 167, 168, 169, 170, 171, 173, 174,
 191,  197, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 215,
 218, 220, 225, 226, 227, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388,
 389, 390, 391, 392, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427,
 428]

#neurons = [56, 57, 58, 59, 61, 64, 68, 69, 72, 73, 75, 78, 80, 133, 134, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 159, 161, 163, 164, 167, 168, 169, 170, 171, 173, 174, 208, 209, 210, 215, 218, 220, 225, 226, 227, 387, 388, 389, 390, 391, 392, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428]
#neurons = [68]
neurons = range(1, 429)
c = 0 
sig_cells = []
for nid in neurons: #sorted(map(str, [111, 109, 423, 128, 108, 123, 58, 61, 72, 190, 405, 402, 187])):
	stats = []
	shuffle_ids = []
	files = [x for x in sorted(glob.glob(f"stats/{nid}_Allo*"))]
	files = sorted(glob.glob(f"stats/{nid}_Ego*CV*"))

	#print(nid, len(files))
	# print(files)
	if len(files) == 0:
		print(nid, "zero files")
		continue
	for f in files:
		dict1 = {}
		d = open(f).read().splitlines()
		# shuffle_ids.append( int(f.split('_')[-1].split('.')[0]) )
		shuffle_ids.append( int(f.split('_')[-2].split('.')[0]) )
		for i, l in enumerate(d):
			if i == 0: continue
			key, value = l.split(": ")
			dict1[key] = float(value)

		if dict1['DIC'] > 10**6:
			print(f)
		else:
			stats.append(dict1)

	to_plot = ['R_spearman', 'R', 'MSE', 'DIC', 'AIC', 'MPD']	
	fig, ax = plt.subplots(2, 3)
	axis = [ax[0][0], ax[0][1], ax[1][0], ax[1][1], ax[0][2], ax[1][2]]

	plt.suptitle(f"Neuron {nid} - #{len(stats)}")

	stats = [x for _, x in sorted(zip(shuffle_ids, stats))]

	for i, x in enumerate(to_plot):
		l = []
		for s in stats:
			l.append(s[x])

		counts, bins = np.histogram(l[1:], bins=20)
		axis[i].hist(bins[:-1], bins, weights=counts / max(counts))
		g = stats_lib.norm.pdf(bins, np.mean(l[1:]), np.std(l[1:]))
		g /= max(g)
		axis[i].plot(bins, g)

		z = (l[0] - np.mean(l[1:])) / np.std(l[1:])

		if x in ['R_spearman', 'R']:
			p_val = 1 - stats_lib.norm.cdf(z)
		else:
			p_val = stats_lib.norm.cdf(z)

		if x == 'R_spearman' and p_val < 0.05:
			c += 1
			sig_cells.append(nid)
			print(c, "-", nid)

		axis[i].scatter(l[0], 0, color='red')
		axis[i].axvline(l[0], 0, 1, color='black',linestyle='solid')
		axis[i].set_title(f"{x} - {p_val:.2}")

	plt.tight_layout()	
	plt.savefig(rf"C:\tmp\new_shuffles\{nid}_Ego.png") #shuffles_dist_ego\{nid}.png")
	# plt.show()

print("SIG_CELLS:", sig_cells)