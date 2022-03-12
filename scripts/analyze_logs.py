import re
import pandas as pd

LOG_PATH = "../logs/sim_logs.txt"
DIC_LOG_PATH = "../logs/sim_dic.txt"

LOG_PATH = "../logs/log5.txt"
DIC_LOG_PATH = "../logs/dic_log5.txt"

dic = {}
aic = {}
n_spikes = {}
d = open(LOG_PATH, "r").read().splitlines()
d2 = d[:]
for l in range(len(d)):
    if d[l] in ['File not found', 'Too few spikes']:
        d2[l] = ''
        d2[l-1] = ''

d3 = "\n".join(d2).replace("\n\n\n", "\n").splitlines()
last_nid = None
import pprint
pprint.pprint(d)
mse = {}
for l in range(len(d3)):
    if 'Loading Data' in d3[l]:
        nid = re.findall('([\\d]+)', d3[l])[0]
        last_nid = nid
        # print(nid)
    if d3[l].startswith("R: 0."):
        # print(d3[l])
        if last_nid in dic:
            dic[last_nid] = (dic[last_nid], d3[l])
        else:
            dic[last_nid] = d3[l]
    if 'AIC:' in d3[l]:
        if last_nid in aic:
            aic[last_nid] = (aic[last_nid], d3[l])
        else:
            aic[last_nid] = d3[l]
    if 'no.spikes:' in d3[l]:
        n_spikes[nid] = d3[l]
    if 'MSE' in d3[l]:
        if last_nid in mse:
            mse[last_nid] = (mse[last_nid], d3[l])
        else:
            mse[last_nid] = d3[l]

def neuron_id_to_day(neuron_id):
    if 1 <= neuron_id <= 29: return "d191222"
    if 31 <= neuron_id <= 53: return "d191221"
    if 56 <= neuron_id <= 80: return "d191220"
    if 82 <= neuron_id <= 104: return "d191223"
    if 106 <= neuron_id <= 129: return "d191224"
    if 132 <= neuron_id <= 144: return "d191225"
    if 145 <= neuron_id <= 161: return "d191226"
    if 163 <= neuron_id <= 174: return "d191229"
    if 176 <= neuron_id <= 190: return "d191231"
    if 191 <= neuron_id <= 207: return "d200101"
    if 208 <= neuron_id <= 216: return "d200102"
    if 217 <= neuron_id <= 227: return "d200108"
    if 228 <= neuron_id <= 233: return "d190603"
    if 236 <= neuron_id <= 244: return "d190604"
    if 246 <= neuron_id <= 252: return "d190610"
    if 254 <= neuron_id <= 265: return "d190612"
    if 270 <= neuron_id <= 272: return "d190617"
    if 274 <= neuron_id <= 280: return "d190924"
    if 282 <= neuron_id <= 291: return "d190925"
    if 292 <= neuron_id <= 298: return "d190926"
    if 300 <= neuron_id <= 339: return "d190928"
    if 375 <= neuron_id <= 377: return "d200419"
    if 378 <= neuron_id <= 380: return "d200420"
    if 381 <= neuron_id <= 386: return "d200421"
    if 387 <= neuron_id <= 389: return "d200422"
    if 390 <= neuron_id <= 392: return "d200423"
    if 393 <= neuron_id <= 394: return "d200425"
    if 395 <= neuron_id <= 400: return "d200426"
    if 401 <= neuron_id <= 406: return "d200427"
    if 407 <= neuron_id <= 415: return "d200428"
    if 416 <= neuron_id <= 423: return "d200429"
    if 424 <= neuron_id <= 428: return "d200430"

def neuron_to_name(nid):
    nid = int(nid)
    if nid < 1000:
        day = neuron_id_to_day(nid)
        return day, nid
    file_id = (nid % 1000)
    files = ['A_place_cell.csv', 'ego_cell1.csv', 'ego_cell2.csv', 'ego_cell3.csv', 'ego_cell4.csv', 'ego_cell_1_2.csv', 'HD_place_cell.csv', 'HD_place_cell_pos1.csv', 'pairwise_distance1,2_cell.csv', 'pairwise_distance1,3_cell.csv', 'pairwise_distance1,4_cell.csv', 'pairwise_distance2,3_cell.csv', 'pairwise_distance2,4_cell.csv', 'pairwise_distance3,4_cell.csv', 'place_distance1_cell.csv', 'place_distance2_cell.csv', 'place_distance3_cell.csv', 'place_distance4_cell.csv', 'randomly_firing_cell.csv']
    if file_id >= len(files):
        # print("ERR")
        return None, None
    day = 191220 + (nid // 1000 - 1)
    return f"d{day}", f"{files[file_id][:-4]}", #_{nid}"

df = pd.DataFrame(columns=['day', 'neuron name', 'n_spikes', 'model type', 'R', 'MSE', 'pDIC', 'DIC', 'AIC'])
d4 = open(DIC_LOG_PATH, "r").read().splitlines()
i = 0
for l in d4:
    if 'None' not in l and 'models' in l:
        nid = l.split(' ')[0]
        day, neuron_name = neuron_to_name(nid)
        if neuron_name:
            model_type = ["ALLO", "EGO"][i % 2]
            if nid not in dic:
                continue
            print(nid, dic[nid][i % 2])
            R_per_model = dic[nid][i % 2].split(' ')[-1]
            aic_per_model = aic[nid][i % 2].split(' ')[-1]
            pdic, dic_score = l.split(' (')[-1][:-1].split(', ')
            n_spikes_val = n_spikes[nid].split(' ')[-1]
            mse_per_model = mse[nid][i % 2].split(' ')[-1]
            row = pd.Series([day, neuron_name, n_spikes_val, model_type, R_per_model, mse_per_model, pdic, dic_score, aic_per_model], index=df.columns)
            df = df.append(row, ignore_index=True)
        i += 1
delta_DIC = df['DIC'][::2].reset_index().astype('float') - df['DIC'][1::2].reset_index().astype('float') 
df['delta_DIC'] = ((delta_DIC['DIC']).repeat(2).reset_index())['DIC']
delta_AIC = df['AIC'][::2].reset_index().astype('float') - df['AIC'][1::2].reset_index().astype('float') 
df['delta_AIC'] = ((delta_AIC['AIC']).repeat(2).reset_index())['AIC']
delta_MSE = df['MSE'][::2].reset_index().astype('float') - df['MSE'][1::2].reset_index().astype('float') 
df['delta_MSE'] = ((delta_MSE['MSE']).repeat(2).reset_index())['MSE']
delta_R = df['R'][::2].reset_index().astype('float') - df['R'][1::2].reset_index().astype('float') 
df['delta_R'] = ((delta_R['R']).repeat(2).reset_index())['R']

df.to_csv("../logs/logs5.csv")
#print(d3)
# print(re.findall('(?!(Loading Data \[\d+\]...\\nFile not found).)*', d))

# scatter plots
import matplotlib.pyplot as plt
import numpy as np
d_dic = df[::2]['delta_DIC'].reset_index()
dic_allo = df[::2]['DIC'].astype('float').reset_index()
plt.scatter(x=np.abs(d_dic['delta_DIC']),y=dic_allo['DIC'])
plt.xlabel("deltaDIC")
plt.ylabel("DIC allo")
#plt.show()

plt.hist(np.abs(d_dic['delta_DIC']))
plt.title("Hist of delta DIC")
#plt.show()


plt.scatter(x=df['delta_DIC'],y=df['delta_AIC'])
plt.xlabel("delta_DIC")
plt.ylabel("delta_AIC")
#plt.show()

cond1 = df['delta_MSE'].astype('float') < 0 # allo better than ego
cond2 = df['delta_R'].astype('float') > 0 # allo better than ego
cond3 = df['delta_DIC'].astype('float').abs() > 10
cond4 = df['delta_DIC'].astype('float') < 0 # allo better than ego
cond5 = df['delta_AIC'].astype('float') < 0 # allo better than ego
cri1 = (cond1 & cond2 & cond3 & cond4 & cond5)
cri2 = (~cond1 & ~cond2 & cond3 & ~cond4 & ~cond5)
print(cri1.mean(), "Allo")
print(cri2.mean(), "Ego")
print((cri1 | cri2).mean(), "Both")