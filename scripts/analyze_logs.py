import re
import pandas as pd

LOG_PATH = "../logs/log4.txt"
DIC_LOG_PATH = "../logs/dic_log4.txt"
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

def neuron_to_name(nid):
    nid = int(nid)
    if nid < 1000:
        day = neuron_to_name(nid)
        return day, nid
    file_id = (nid % 1000)
    files = ['A_place_cell.csv', 'ego_cell1.csv', 'ego_cell2.csv', 'ego_cell3.csv', 'ego_cell4.csv', 'ego_cell_1_2.csv', 'HD_place_cell.csv', 'HD_place_cell_pos1.csv', 'pairwise_distance1,2_cell.csv', 'pairwise_distance1,3_cell.csv', 'pairwise_distance1,4_cell.csv', 'pairwise_distance2,3_cell.csv', 'pairwise_distance2,4_cell.csv', 'pairwise_distance3,4_cell.csv', 'place_distance1_cell.csv', 'place_distance2_cell.csv', 'place_distance3_cell.csv', 'place_distance4_cell.csv', 'randomly_firing_cell.csv']
    if file_id >= len(files):
        # print("ERR")
        return None, None
    day = 191220 + (nid // 1000 - 1)
    return f"d{day}", f"{files[file_id][:-4]}", #_{nid}"

df = pd.DataFrame(columns=['day', 'neuron name', 'n_spikes', 'model type', 'R', 'pDIC_ps', 'DIC_ps', 'pDIC', 'DIC', 'AIC'])
d4 = open(DIC_LOG_PATH, "r").read().splitlines()
i = 0
for l in d4:
    if 'None' not in l and 'models' in l:
        nid = l.split(' ')[0]
        day, neuron_name = neuron_to_name(nid)
        if neuron_name:
            model_type = ["ALLO", "EGO"][i % 2]
            if nid not in dic:
                break
            R_per_model = dic[nid][i % 2].split(' ')[-1]
            aic_per_model = aic[nid][i % 2].split(' ')[-1]
            pdic_ps, dic_score_ps = l.split(' (')[-1][:-1].split(', ')
            n_spikes_val = n_spikes[nid].split(' ')[-1]
            pdic, dic_score = float(pdic_ps) * int(n_spikes_val), float(dic_score_ps) * int(n_spikes_val)
            print(neuron_name, model_type, n_spikes_val, R_per_model, pdic_ps, dic_score_ps, aic_per_model)
            row = pd.Series([day, neuron_name, model_type,n_spikes_val, R_per_model, pdic_ps, dic_score_ps, pdic, dic_score, aic_per_model], index=df.columns)
            df = df.append(row, ignore_index=True)
        i += 1
df.to_csv("../logs/logs2.csv")
#print(d3)
# print(re.findall('(?!(Loading Data \[\d+\]...\\nFile not found).)*', d))