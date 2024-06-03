import numpy as np
import random
import math


def pretreat_oridata(tmp_oridata):
    sum_ad = sum(tmp_oridata[:9])
    sum_dis = sum(tmp_oridata[9:19])
    sum_shape = sum(tmp_oridata[19:])
    if sum_ad != 0:
        for i in range(9):
            tmp_oridata[i] = float(tmp_oridata[i] / sum_ad)
    if sum_dis != 0:
        for i in range(9, 19):
            tmp_oridata[i] = float(tmp_oridata[i] / sum_dis)
    if sum_shape != 0:
        for i in range(19, 27):
            tmp_oridata[i] = float(tmp_oridata[i] / sum_shape)
    return tmp_oridata


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def position_encoding(pos, d_model):
    div_term = np.exp(np.arange(0, d_model, 2).astype(
        np.float32) * (-math.log(10000.0) / d_model))
    position = np.zeros(d_model)
    position[0::2] = np.sin(pos * div_term)
    position[1::2] = np.cos(pos * div_term)
    return position


def getdict():
    dict_seq = {}
    dict_pdb = {}
    nRows = 3
    i = 0
    lines = []
    for line in open("data/pdblabel.txt"):
        i = i + 1
        if i <= nRows:
            lines.append(line.replace('\n', ''))
        else:
            pdb_id = lines[0].split(">")[1][:4].lower()
            chain_id = lines[0].split(">")[1][4:]
            for chain in chain_id:
                uniprot_id = pdb_id + "_" + chain
                dict_pdb.update({uniprot_id: lines[2]})
                dict_seq.update({uniprot_id: lines[1]})
            lines = []
            lines.append(line.replace('\n', ''))
            i = 1
    if lines != []:
        pdb_id = lines[0].split(">")[1][:4].lower()
        chain_id = lines[0].split(">")[1][4:]
        for chain in chain_id:
            uniprot_id = pdb_id + "_" + chain
            dict_pdb.update({uniprot_id: lines[2]})
            dict_seq.update({uniprot_id: lines[1]})
        lines = []
        lines.append(line.replace('\n', ''))
        i = 1

    return dict_seq, dict_pdb


def cal_sum(num):
    sum = 0
    for i in range(len(num)):
        sum = sum + num[i]
    return sum


def cal_mean(num):
    return cal_sum(num) / len(num)


def cal_ad(num):
    return cal_sum(np.abs(num - cal_mean(num))) / len(num)


def tenTotwo(number):
    s = [0] * 5
    i = 4
    while number > 0:
        rem = number % 2
        s[i] = rem
        i -= 1
        number = number // 2
    return s

def generatematrix(pdbid, dict_pdb, dict_seq):
    codes = {'A': 0, 'R': 1, 'N': 2, 'D': 3,
             'C': 4, 'Q': 5, 'E': 6, 'G': 7,
             'H': 8, 'I': 9, 'L': 10, 'K': 11,
             'M': 12, 'F': 13, 'P': 14, 'S': 15,
             'T': 16, 'W': 17, 'V': 18, 'Y': 19}
    chain = pdbid.split("_")[1]
    pdbid = pdbid[0:4].lower()+'_'+chain
    input_label = dict_pdb.get(pdbid)
    seq = dict_seq.get(pdbid)
    try:
        n = len(input_label)
    except:
        print(pdbid)
    try:
        pssm = np.load("features/pssm_large/" +
                       pdbid.split("_")[0]+chain + ".npy")
    except:
        pssm = np.load("features/pssm_large/" + pdbid.split("_")
                       [0].upper()+chain + ".npy")

    try:
        dssp = np.load("features/dssp/" + pdbid.split("_")[0]+chain + ".npy")
    except:
        dssp = np.load("features/dssp/" + pdbid.split("_")
                       [0].upper()+chain + ".npy")

    try:
        hmm = np.load("features/hmm/" + pdbid.split("_")[0]+chain + ".npy")
    except:
        hmm = np.load("features/hmm/" + pdbid.split("_")
                      [0].upper()+chain + ".npy")

    pos = []
    raw_protein = []
    for i in range(len(seq)):
        protein = seq[i]
        tmp = np.zeros(20)
        tmp[int(codes.get(protein))] = 1
        raw_protein.append(tmp)
        pos.append(position_encoding(i, 20))

    _raw_protein = []
    for i in range(len(raw_protein)):
        _raw_protein.append(raw_protein[(i-1) % len(raw_protein)]+2*raw_protein[(
            i) % len(raw_protein)]+3*raw_protein[(i+1) % len(raw_protein)])
    raw_protein = _raw_protein

    oridata = []
    tmpseq = ""
    num = -1
    tmp_oridata = [0] * 28
    dis_map = [[]]
    try:
        fp_ori = open("features/oridata_dismap/"+pdbid+".data")
    except:
        fp_ori = open("features/oridata_dismap/"+pdbid.upper()+".data")
    for line in fp_ori:
        npdata = line.strip().split("\t")
        _num, _seq = npdata[4].split("_")
        ad = float(npdata[0])
        dis = float(npdata[1])
        K = float(npdata[2])
        H = float(npdata[3])
        loc_str = npdata[5].strip("[]").split(" ")
        loc = []
        for coordinate in loc_str:
            if coordinate != "":
                loc.append(float(coordinate))
        dis_map[-1].append(loc)

        if ad < -0.5:
            tmp_oridata[0] += 1
        elif -0.5 <= ad < -0.3:
            tmp_oridata[1] += 1
        elif -0.3 <= ad < -0.1:
            tmp_oridata[2] += 1
        elif -0.1 <= ad < -0.001:
            tmp_oridata[3] += 1
        elif -0.001 <= ad <= 0.001:
            tmp_oridata[4] += 1
        elif 0.001 < ad <= 0.1:
            tmp_oridata[5] += 1
        elif 0.1 < ad <= 0.3:
            tmp_oridata[6] += 1
        elif 0.3 < ad <= 0.5:
            tmp_oridata[7] += 1
        else:
            tmp_oridata[8] += 1

        if -0.2 <= dis < 0:
            tmp_oridata[9] += 1
        elif -0.6 <= dis < -0.2:
            tmp_oridata[10] += 1
        elif -1.2 <= dis < -0.6:
            tmp_oridata[11] += 1
        elif -2.0 <= dis < -1.2:
            tmp_oridata[12] += 1
        elif dis < -2.0:
            tmp_oridata[13] += 1
        elif dis > 2.0:
            tmp_oridata[14] += 1
        elif 2.0 >= dis > 1.2:
            tmp_oridata[15] += 1
        elif 1.2 >= dis > 0.6:
            tmp_oridata[16] += 1
        elif 0.6 >= dis > 0.2:
            tmp_oridata[17] += 1
        else:
            tmp_oridata[18] += 1

        if K == 0 and H > 0:
            tmp_oridata[19] += 1
        if K == 0 and H == 0:
            tmp_oridata[20] += 1
        if K == 0 and H < 0:
            tmp_oridata[21] += 1

        if K < 0 and H > 0:
            tmp_oridata[22] += 1
        if K < 0 and H == 0:
            tmp_oridata[23] += 1
        if K < 0 and H < 0:
            tmp_oridata[24] += 1

        if K > 0 and H > 0:
            tmp_oridata[25] += 1
        if K > 0 and H < 0:
            tmp_oridata[26] += 1
        if _num != num:
            dis_map[-1] = np.mean(dis_map[-1], axis=0).tolist()
            dis_map.append([])
            num = _num
            oridata.append(pretreat_oridata(tmp_oridata))
            tmp_oridata = [0] * 28
            tmpseq += _seq
            num = _num
    oridata.append(pretreat_oridata(tmp_oridata))
    tmpseq += _seq
    if dis_map[-1] != []:
        dis_map[-1] = np.mean(dis_map[-1], axis=0).tolist()
    i = 0
    position = tmpseq.find(seq[i: i + 3])
    while position == -1:
        i += 1
        position = tmpseq.find(seq[i: i + 3])
    _tmpseq = ""
    pointer_a = 0
    pointer_b = 0
    _oridata = []
    _dis_map = []
    lossamino = 0
    while pointer_a < len(tmpseq) and pointer_b < n:
        if tmpseq[pointer_a] != seq[pointer_b]:
            _tmpseq += "*"
            lossamino += 1
            _oridata.append([0] * 27 + [1])
            _dis_map.append(np.mean(
                [dis_map[pointer_a], dis_map[(pointer_a + 1) % len(tmpseq)]], axis=0).tolist())
            pointer_b += 1
        else:
            _tmpseq += tmpseq[pointer_a]
            _oridata.append(oridata[pointer_a])
            _dis_map.append(dis_map[pointer_a])
            pointer_a += 1
            pointer_b += 1

    if len(_tmpseq) != n:
        for i in range(n - len(_tmpseq)):
            _oridata.append([0] * 27 + [1])
            _dis_map.append(np.mean(
                [dis_map[pointer_a - 1], dis_map[(pointer_a) % len(tmpseq)]], axis=0).tolist())
        _tmpseq += "*" * (n - len(_tmpseq))
        lossamino += n - len(_tmpseq)
    
    input_matrix = np.concatenate((pssm, hmm), axis=-1)
    input_matrix = np.concatenate((input_matrix, raw_protein), axis=-1)
    input_matrix = np.concatenate((input_matrix, pos), axis=-1)
    input_matrix = np.concatenate((input_matrix, dssp), axis=-1)
    input_matrix = np.concatenate((input_matrix, _oridata), axis=-1)
    
    np.save("features/matrix/"+pdbid+"_input.npy", input_matrix)
    np.save("features/matrix/"+pdbid+"_label.npy", input_label)
    
    return input_matrix, input_label

'''
dict_seq, dict_pdb = getdict()
# Legitimacy_check(dict_pdb, dict_seq)
i = 0
result = []
result_shape = []
for line in open("data/eval.txt"):
    # if i < 273:
    #    i+=1
    #    continue
    pdbid = line.strip()
    input_matrix, input_label, lossamino = generatematrix(
        pdbid, dict_pdb, dict_seq)
    concave_index = 0
    for i in range(len(input_label)):
        for score in input_matrix[i]:
            concave_index += abs(float(score))
    print(f"{pdbid}\t {lossamino}\t {len(input_label)}\t {concave_index}")
for line in open("data/train.txt"):
    # if i < 273:
    #    i+=1
    #    continue
    pdbid = line.strip()
    input_matrix, input_label= generatematrix(
        pdbid, dict_pdb, dict_seq)
    for i in range(len(input_label)):
        if input_label[i] == "1":
            result.append(input_matrix[i][94: 103])
            result_shape.append(input_matrix[i][113:121])
result = np.array(result)
result_shape = np.array(result_shape)
'''
# np.save("cc_feature.npy", result.T)
# np.save("shape_feature.npy", result_shape.T)
