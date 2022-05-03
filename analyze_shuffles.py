import glob
import matplotlib.pyplot as plt

nid = 72

rs = []
mses = []
stats = []

files = glob.glob(f"stats/{nid}*")
for f in files:
	dict1 = {}
	d = open(f).read().splitlines()
	for i, l in enumerate(d):
		if i == 0: continue
		key, value = l.split(": ")
		dict1[key] = float(value)
	stats.append(dict1)

to_plot = ['R', 'MSE', 'DIC', 'AIC']
fig, ax = plt.subplots(2, 2)
axis = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]

plt.suptitle(f"Neuron {nid}")

for i, x in enumerate(to_plot):
	l = []
	for s in stats:
		l.append(s[x])


	axis[i].hist(l[1:])
	axis[i].scatter(l[0], 0, color='red')
	axis[i].set_title(x)

plt.show()

