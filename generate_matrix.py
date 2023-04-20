import numpy as np
from itertools import islice
import re
import os
import random
import math

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
 
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def position_encoding(pos, d_model):
    div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * (-math.log(10000.0) / d_model))
    position = np.zeros(d_model)
    position[0::2] = np.sin(pos * div_term)
    position[1::2] = np.cos(pos * div_term)
    return position

def getdict():
    dict_seq = {}
    dict_pdb = {}
    dict_lab = {}
    dict_Uniprot = {}
    nRows = 3
    i = 0
    lines = []
    for line in open("pdblabel.txt"):
        i = i + 1
        if i <= nRows:
            lines.append(line.replace('\n',''))
        else:
            pdb_id = lines[0].split(">")[1][:4].lower()
            chain_id = lines[0].split(">")[1][4:]
            for chain in chain_id:
                uniprot_id = pdb_id + "_" + chain
                dict_pdb.update({uniprot_id: lines[2]})
                dict_seq.update({uniprot_id: lines[1]})
            lines = []
            lines.append(line.replace('\n',''))
            i = 1
    if lines != []:
        pdb_id = lines[0].split(">")[1][:4].lower()
        chain_id = lines[0].split(">")[1][4:]
        for chain in chain_id:
            uniprot_id = pdb_id + "_" + chain
            dict_pdb.update({uniprot_id: lines[2]})
            dict_seq.update({uniprot_id: lines[1]})
        lines = []
        lines.append(line.replace('\n',''))
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
    '''
    try:
        input_pssm = open("./pssm/"+pdbid+".pssm")
    except:
        input_pssm = open("./pssm/"+pdbid.upper()+".pssm")
    pssm = []
    seq_pssm = ""
    for line in islice(input_pssm,3,None):
        try:
            protein = re.findall('[A-Z]',line)
            tmp_pssm = []
            for num in line.split(protein[0])[1].split(' '):
                if num != '' and len(tmp_pssm) < 20:
                    tmp_pssm.append(num)
            tmp_pssm = np.array(tmp_pssm).astype(float)
            seq_pssm += protein[0]
            pssm.append(tmp_pssm)
        except:
            break
    '''
    try:
        pssm = np.load("pssm_large/" + pdbid.split("_")[0]+chain + ".npy")
    except:
        pssm = np.load("pssm_large/" + pdbid.split("_")[0].upper()+chain + ".npy")
    
    try:
        dssp = np.load("dssp/" + pdbid.split("_")[0]+chain + ".npy")
    except:
        dssp = np.load("dssp/" + pdbid.split("_")[0].upper()+chain + ".npy")
    
    try:
        hmm = np.load("hmm/" + pdbid.split("_")[0]+chain + ".npy")
    except:
        hmm = np.load("hmm/" + pdbid.split("_")[0].upper()+chain + ".npy")
 
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
        _raw_protein.append(raw_protein[(i-1) % len(raw_protein)]+2*raw_protein[(i) % len(raw_protein)]+3*raw_protein[(i+1) % len(raw_protein)])
    raw_protein = _raw_protein

    oridata = []
    tmpseq = ""
    num = -1
    tmp_oridata = [0] * 28
    dis_map = [[]]
    try:
        fp_ori = open("oridata_dismap/"+pdbid+".data")
    except:
        fp_ori = open("oridata_dismap/"+pdbid.upper()+".data")
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
        elif dis > 2.0 :
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
            oridata.append(tmp_oridata)
            tmp_oridata = [0] * 28
            tmpseq += _seq
            num = _num
    oridata.append(tmp_oridata)        
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
    while pointer_a < len(tmpseq) and pointer_b < n:
        if tmpseq[pointer_a] != seq[pointer_b]:
            _tmpseq += "*"
            _oridata.append([0] * 27 + [1])
            _dis_map.append(np.mean([dis_map[pointer_a], dis_map[(pointer_a + 1) % len(tmpseq)]], axis=0).tolist())
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
            _dis_map.append(np.mean([dis_map[pointer_a - 1], dis_map[(pointer_a) % len(tmpseq)]], axis=0).tolist())
        _tmpseq += "*" * (n - len(_tmpseq))
    dis_map = _dis_map
    lenOfmap = len(dis_map) 
    all_map = [[np.inf] * lenOfmap for _ in range(lenOfmap)]
    for i in range(0, lenOfmap):
        for j in range(0, lenOfmap):
            if i == j:
                all_map[i][j] = 0
                continue
            if dis_map[i][0] == np.inf or dis_map[j][0] == np.inf:
                continue
            x = dis_map[j][0] - dis_map[i][0]
            y = dis_map[j][1] - dis_map[i][1]
            z = dis_map[j][2] - dis_map[i][2]
            all_map[i][j] = pow(x+y+z, 2)
    all_map = np.array(all_map)
#    print(seq)
#    print(_tmpseq)
#    pssm = standardization(normalization(pssm))
#    dssp = standardization(normalization(dssp))
#    _oridata = standardization(normalization(_oridata))
    #print(len(pssm[0]), len(hmm[0]), len(dssp[0]), len(_oridata[0]), len(feature_phy_char[0]), len(feature_phy_prop[0]), len(pos[0]))
    input_matrix = np.concatenate((pssm, hmm), axis= -1)
    input_matrix = np.concatenate((input_matrix, raw_protein), axis=-1)
    input_matrix = np.concatenate((input_matrix, pos), axis=-1)
    input_matrix = np.concatenate((input_matrix, dssp), axis=-1)
    input_matrix = np.concatenate((input_matrix, _oridata), axis=-1)
    '''
    input_matrix_1 = []
    input_label_1 = ''
    for i in range(n):
        if _tmpseq[i] != '*':
            input_matrix_1.append(input_matrix[i])
            input_label_1 += input_label[i]
    '''
    return input_matrix, input_label, all_map

def Legitimacy_check( dict_pdb, dict_seq):
    orddir = open('data/Dset_all.txt')
    numpdb = 0
    list_0 = []
    list_1 = []
    list_02 = []
    list_24 = []
    list_46 = []
    list_68 = []
    list_80 = []
    sum_0, sum_1 = 0, 0
    for pdb in orddir:
        try:
            input_label = dict_pdb.get(pdb.strip())
            array = np.array(list(input_label))
            num0 = np.where(array == '0')[0].shape[0]
            num1 = np.where(array == '1')[0].shape[0]
            sum_0 += num0
            sum_1 += num1
            numpdb += 1
            if num0 == 0:
                list_1.append(pdb.split('.')[0])
            elif num1 == 0:
                list_0.append(pdb.split('.')[0])
            elif num0 != 0 and num1 != 0 and float(num0 / (num1 + num0)) > 0 and float(num0 / (num1 + num0)) <= 0.2:   
                list_02.append(pdb.split('.')[0])
            elif num0 != 0 and num1 != 0 and float(num0 / (num1 + num0)) > 0.2 and float(num0 / (num1 + num0)) <= 0.4:   
                list_24.append(pdb.split('.')[0])
            elif num0 != 0 and num1 != 0 and float(num0 / (num1 + num0)) > 0.4 and float(num0 / (num1 + num0)) <= 0.6:   
                list_46.append(pdb.split('.')[0])
            elif num0 != 0 and num1 != 0 and float(num0 / (num1 + num0)) > 0.6 and float(num0 / (num1 + num0)) <= 0.8:   
                list_68.append(pdb.split('.')[0])
            elif num0 != 0 and num1 != 0 and float(num0 / (num1 + num0)) > 0.8 and float(num0 / (num1 + num0)) <= 1:   
                list_80.append(pdb.split('.')[0])
        except:
            numpdb += 1
            with open('Defectivepdb','a+') as f:
                f.write(pdb.split('.')[0]+'\n')
    list_len = [len(list_0), len(list_1), len(list_02), len(list_24), len(list_46), len(list_68), len(list_80)]
    allpdb = sum_0 + sum_1 
    print(sum_0, sum_1, float(allpdb / sum_0), float(allpdb / sum_1))
    random.shuffle(list_0)
    random.shuffle(list_1)
    random.shuffle(list_02)
    random.shuffle(list_24)
    random.shuffle(list_46)
    random.shuffle(list_68)
    random.shuffle(list_80)
    list_all = list_0 + list_1 + list_02 + list_24 + list_46 + list_68 + list_80 

    list_train = []
    list_eval = []
    f1 = open('data/train.txt','a+')
    f2 = open('data/eval.txt','a+')
    num_train, num_eval = 0, 0
    for i in range(len(list_all)):
        if i % 5 != 4:
            list_train.append(list_all[i])# + list_all[i] if list_all[i] not in list_80 else list_all[i])
        else:
            list_eval.append(list_all[i])
    random.shuffle(list_train)
    for i in range(len(list_train)):        
        f1.write(list_train[i])
    random.shuffle(list_eval)
    for i in range(len(list_eval)):
        f2.write(list_eval[i])

dict_seq, dict_pdb= getdict()
#Legitimacy_check(dict_pdb, dict_seq)
'''
i = 0
with open(os.path.join('data', "train.txt"), "r", encoding="utf-8") as f:
    info_train = f.readlines()

with open(os.path.join('data', "eval.txt"), "r", encoding="utf-8") as f:
    info_eval = f.readlines()
sum_1 = 0
sum_0 = 0
num_1 = 0
num_0 = 0
for i in range(len(info_train)):
    pdb_train_id = info_train[i].strip()
    input_label = dict_pdb.get(pdb_train_id)
    lab = np.array(list(input_label))
    tmp_1 = len(np.where(lab == '1')[0])
    tmp_0 = len(np.where(lab == '0')[0])
    num_1 += tmp_1
    num_0 += tmp_0
print(num_1, num_0)
print(num_1/(num_0 + num_1), num_0/(num_0 + num_1))
sum_1 += num_1
sum_0 += num_0
num_1 = 0
num_0 = 0
for i in range(len(info_eval)):
    pdb_train_id = info_eval[i].strip()
    input_label = dict_pdb.get(pdb_train_id)
    lab = np.array(list(input_label))
    tmp_1 = len(np.where(lab == '1')[0])
    tmp_0 = len(np.where(lab == '0')[0])
    num_1 += tmp_1
    num_0 += tmp_0
print(num_1, num_0)
print(num_1/(num_0 + num_1), num_0/(num_0 + num_1))
sum_1 += num_1
sum_0 += num_0
print(sum_1, sum_0)
print(sum_1/(sum_0 + sum_1), sum_0/(sum_0 + sum_1))
i = 0 
for line in open("data/eval.txt"):
    #if i < 273:
    #    i+=1
    #    continue
    pdbid = line.strip()
    input_matrix, input_label, dis_map= generatematrix(pdbid, dict_pdb, dict_seq)
    #j = -1
    #print(input_matrix[j,74:96])
    indices = []
    lenOfmap = len(dis_map)
    for j in range(lenOfmap):
        sorted_indices = np.argsort(dis_map[j])
        print(sorted_indices)
    i += 1
    if i > 0:
        break

'''
