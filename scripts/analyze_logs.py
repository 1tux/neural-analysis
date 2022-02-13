import re

dic = {}

d = open("../log3.txt", "r").read().splitlines()
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

def neuron_to_name(nid):
    nid = int(nid)
    if nid < 1000:
        return nid
    file_id = (nid % 1000)
    files = ['A_place_cell.csv', 'ego_cell1.csv', 'ego_cell2.csv', 'ego_cell3.csv', 'ego_cell4.csv', 'ego_cell_1_2.csv', 'HD_place_cell.csv', 'HD_place_cell_pos1.csv', 'pairwise_distance1,2_cell.csv', 'pairwise_distance1,3_cell.csv', 'pairwise_distance1,4_cell.csv', 'pairwise_distance2,3_cell.csv', 'pairwise_distance2,4_cell.csv', 'pairwise_distance3,4_cell.csv', 'place_distance1_cell.csv', 'place_distance2_cell.csv', 'place_distance3_cell.csv', 'place_distance4_cell.csv', 'randomly_firing_cell.csv']
    if file_id >= len(files):
        # print("ERR")
        return
    day = 191220 + nid // 1000
    return f"{day}_{files[file_id][:-4]}_{nid}"

d4 = open("../dic_log3.txt", "r").read().splitlines()
i = 0
for l in d4:
    if 'None' not in l and 'models' in l:
        nid = l.split(' ')[0]
        neuron_name = neuron_to_name(nid)
        if neuron_name:
            print(neuron_name, ["ALLO", "EGO"][i % 2] ,dic[nid][i % 2].split(' ')[-1], l.split(' (')[-1][:-1])
        i += 1
#print(d3)
# print(re.findall('(?!(Loading Data \[\d+\]...\\nFile not found).)*', d))