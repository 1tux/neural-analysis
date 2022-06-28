import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

LOG_PATH = "../logs/72rs__.txt"

rs = []
dics = []
d = open(LOG_PATH, "r").read().splitlines() + open("../logs/72rs_.txt", "r").read().splitlines()
for l in d:
    if 'R: 0.' in l:
        r = float(l.split(": ")[1])
        rs.append(r)
    if 'DIC' in l:
        dic = float(l.split(": ")[1])
        dics.append(dic)
#print(dics)
#rs = dics
print(len(rs))
rs_shuffle = rs[:-2]
allo_r = rs[-2]
ego_r = rs[-1]

counts, bins = np.histogram(rs_shuffle, bins=20)
plt.hist(bins[:-1], bins, weights=counts / max(counts))
g = stats.norm.pdf(bins, np.mean(rs_shuffle), np.std(rs_shuffle))
g /= max(g)
plt.plot(bins, g)

#plt.show()
#h = plt.hist(rs_shuffle, bins=30)[0]
plt.plot([allo_r, allo_r], [0, plt.gca().get_ylim()[1]])

z = stats.zscore(rs)[-2] # allo z-score
p_val = 1-stats.norm.cdf(z)
plt.title(f"Z-score: {z:.3}; p_value {p_val:.4}")
plt.savefig(f'C:/tmp/cosyne-figs/72_shuffles.pdf', bbox_inches='tight', dpi=300)   
plt.show()