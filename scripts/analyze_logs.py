import re
import pandas as pd
dic = {}

d = open("../logs/log3.txt", "r").read().splitlines()
d2 = d[:]
for l in range(len(d)):
    if d[l] in ['File not found', 'Too few spikes']:
        d2[l] = ''
        d2[l-1] = ''

d3 = "\n".join(d2).replace("\n\n\n", "\n").splitlines()
last_nid = None
for l in range(len(d3)):
    if 'Loading Data' in d3[l]:
        nid = re.findall('([\\d]+)', d3[l])[0]
        last_nid = nid
        # print(nid)
    if 'R 0.' in d3[l]:
        # print(d3[l])
        if last_nid in dic:
            dic[last_nid] = (dic[last_nid], d3[l])
        else:
            dic[last_nid] = d3[l]

# print(dic)

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

df = pd.DataFrame(columns=['day', 'neuron name', 'model type', 'R', 'pDIC', 'DIC'])
d4 = open("../logs/dic_log3.txt", "r").read().splitlines()
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
            pdic, dic_score = l.split(' (')[-1][:-1].split(', ')
            print(neuron_name, model_type , R_per_model, pdic, dic_score)
            row = pd.Series([day, neuron_name, model_type , R_per_model, pdic, dic_score], index=df.columns)
            df = df.append(row, ignore_index=True)
        i += 1
df.to_csv("../logs/logs.csv")
#print(d3)
# print(re.findall('(?!(Loading Data \[\d+\]...\\nFile not found).)*', d))