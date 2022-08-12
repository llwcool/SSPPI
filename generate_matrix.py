import numpy as np
from itertools import islice
import re
import os
import random

def getdict():
    dict_seq = {}
    dict_pdb = {}
    dict_lab = {}
    dict_Uniprot = {}
    nRows = 3
    i = 0
    lines = []
    for line in open("alllabel.txt"):
        i = i + 1
        if i <= nRows:
            lines.append(line.replace('\n',''))
        else:
            uniprot_id = lines[0].split('>')[1]
            dict_lab.update({uniprot_id: lines[2]})
            dict_Uniprot.update({uniprot_id: lines[1]})
            lines = []
            lines.append(line.replace('\n',''))
            i = 1
    if lines != []:
        uniprot_id = lines[0].split('>')[1]
        dict_lab.update({uniprot_id: lines[2]})
        dict_Uniprot.update({uniprot_id: lines[1]})
        lines = []
        lines.append(line.replace('\n',''))
        i = 1

    #O34757 3EHF_A,175-370
    for line in open("pdblist"):
        if line.startswith('-'):
            continue
        elif len(line.split(' ')) > 1:
            pdb_id = line.split(' ')[1].split(',')[0]
            start = int(line.split(' ')[1].split(',')[1].split('-')[0])
            end = int(line.replace('\n','').split(' ')[1].split(',')[1].split('-')[1])
            dict_pdb.update(({pdb_id: dict_lab.get(line.split(' ')[0])[start-1:end]}))
            dict_seq.update(({pdb_id: dict_Uniprot.get(line.split(' ')[0])[start - 1:end]}))
        else:
            pdb_id = line.replace('\n','')
            pdb = pdb_id.split('_')[0]
            chain = pdb_id.split('_')[1]
            dict_pdb.update(({pdb+'_'+chain: dict_lab.get(pdb+chain)}))
            dict_seq.update(({pdb+'_'+chain: dict_Uniprot.get(pdb + chain)}))
    return dict_seq, dict_pdb
'''
dict_pdb contains the label of the pdb
use dict_pdb.get("idofpdb") to get the label
'''

'''
这里获取路径下所有的geometry_data，再结合pssm矩阵生成待训练矩阵
'''
def cal_sum(num):
    sum = 0
    for i in range(len(num)):
        sum = sum + num[i]
    return sum

def cal_mean(num):
    return cal_sum(num) / len(num)

def cal_ad(num):
    return cal_sum(np.abs(num - cal_mean(num))) / len(num)

def generatematrix(pdbid, dict_pdb, dict_seq):

    codes = {'A': 0, 'R': 1, 'N': 2, 'D': 3,
             'C': 4, 'Q': 5, 'E': 6, 'G': 7,
             'H': 8, 'I': 9, 'L': 10, 'K': 11,
             'M': 12, 'F': 13, 'P': 14, 'S': 15,
             'T': 16, 'W': 17, 'V': 18, 'Y': 19}

    input_label = dict_pdb.get(pdbid)
    try:
        n = len(input_label)
    except:
        print(pdbid)
    input_matrxi = np.zeros((n,6,20)).astype(float)
    input_pssm = open("./pssm/"+pdbid+".pssm")
    i = 0
    for line in islice(input_pssm,3,None):
        try:
            protein = re.findall('[A-Z]',line)
            pssm = []
            for num in line.split(protein[0])[1].split(' '):
                if num != '' and len(pssm) < 20:
                    pssm.append(num)
            pssm = np.array(pssm).astype(float)
            input_matrxi[i, 0, :20] = pssm
            tmp = np.zeros(20)
            try:
                tmp[int(codes.get(protein[0]))] = 1
            except:
                continue
            input_matrxi[i, 1, :20] = tmp
            i = i + 1
        except:
            break
    n = i
    input_label = input_label[0:n]
    input_matrxi = input_matrxi[0:n]
    data_matrix = []
    try:
        input_data = open("./oridata/"+pdbid+".data").read().split('\n')[0:-1]
    except:
        print(pdbid)
    for data in input_data:
        data_matrix.append(data.split('\t')[:-1])
    npdata = np.array(data_matrix).astype(float)
    i = 0
    seqdiff = 0
    tmp_geo = []
    preindex = 0
    for line in open("./oridata/"+pdbid+".data"):
        content = line.replace('\n', '').split('\t')
        indexofatom = int(content[-1].split('_')[0])
        if i == 0:
            seqdiff = indexofatom - str(dict_seq.get(pdbid)).find(str(content[-1].split('_')[1]))
            preindex = indexofatom+seqdiff-2
        index = indexofatom + seqdiff - 2
        if index >= n:
            break
        if index != preindex and tmp_geo != []:
            tmp_geo = np.array(tmp_geo)
            list = []
            list.append(tmp_geo[:, 0])
            list.append(tmp_geo[:, 1])
            list.append(tmp_geo[:, 2])
            list.append(tmp_geo[:, 3])
            for j in range(4):
                mean = cal_mean(list[j])
                input_matrxi[preindex, j + 2, 13] = mean
                input_matrxi[preindex, j + 2, 14] = np.max(list[j])
                input_matrxi[preindex, j + 2, 15] = np.min(list[j])
                input_matrxi[preindex, j + 2, 16] = cal_ad(list[j])
                input_matrxi[preindex, j + 2, 17] = np.var(list[j])
                input_matrxi[preindex, j + 2, 18] = np.std(list[j])
                if mean != 0:
                    input_matrxi[preindex, j + 2, 19] = np.std(list[j]) / cal_mean(list[j])
                else:
                    input_matrxi[preindex, j + 2, 19] = 0
            tmp_geo = []
        tmp_geo.append(npdata[i])

        for j in range(len(npdata[i])):
            if npdata[i][j] < -1:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 0] = input_matrxi[indexofatom + seqdiff - 2, j+2, 0] + 1
            elif -1 <= npdata[i][j] < -0.8:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 1] = input_matrxi[indexofatom + seqdiff - 2, j+2, 1] + 1
            elif -0.8 <= npdata[i][j] < -0.6:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 2] = input_matrxi[indexofatom + seqdiff - 2, j+2, 2] + 1
            elif -0.6 <= npdata[i][j] < -0.4:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 3] = input_matrxi[indexofatom + seqdiff - 2, j+2, 3] + 1
            elif -0.4 <= npdata[i][j] < -0.2:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 4] = input_matrxi[indexofatom + seqdiff - 2, j+2, 4] + 1
            elif -0.2 <= npdata[i][j] < 0:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 5] = input_matrxi[indexofatom + seqdiff - 2, j+2, 5] + 1
            elif npdata[i][j] == 0:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 6] = input_matrxi[indexofatom + seqdiff - 2, j+2, 6] + 1
            elif 0 < npdata[i][j] <= 0.2:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 7] = input_matrxi[indexofatom + seqdiff - 2, j+2, 7] + 1
            elif 0.2 <= npdata[i][j] <= 0.4:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 8] = input_matrxi[indexofatom + seqdiff - 2, j+2, 8] + 1
            elif 0.4 <= npdata[i][j] <= 0.6:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 9] = input_matrxi[indexofatom + seqdiff - 2, j+2, 9] + 1
            elif 0.6 <= npdata[i][j] <= 0.8:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 10] = input_matrxi[indexofatom + seqdiff - 2, j+2, 10] + 1
            elif 0.8 < npdata[i][j] <= 1:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 11] = input_matrxi[indexofatom + seqdiff - 2, j+2, 11] + 1
            else:
                input_matrxi[indexofatom + seqdiff - 2, j+2, 12] = input_matrxi[indexofatom + seqdiff - 2, j+2, 12] + 1

        i = i + 1
        preindex = index
    if tmp_geo != []:
        tmp_geo = np.array(tmp_geo)
        list = []
        list.append(tmp_geo[:, 0])
        list.append(tmp_geo[:, 1])
        list.append(tmp_geo[:, 2])
        list.append(tmp_geo[:, 3])
        for j in range(4):
            mean = cal_mean(list[j])
            input_matrxi[preindex, j + 2, 13] = mean
            input_matrxi[preindex, j + 2, 14] = np.max(list[j])
            input_matrxi[preindex, j + 2, 15] = np.min(list[j])
            input_matrxi[preindex, j + 2, 16] = cal_ad(list[j])
            input_matrxi[preindex, j + 2, 17] = np.var(list[j])
            input_matrxi[preindex, j + 2, 18] = np.std(list[j])
            if mean != 0:
                input_matrxi[preindex, j + 2, 19] = np.std(list[j]) / cal_mean(list[j])
            else:
                input_matrxi[preindex, j + 2, 19] = 0
        tmp_geo = []
    return input_matrxi, input_label


def Legitimacy_check( dict_pdb, dict_seq):
    with open('Defectivepdb', 'w') as f:
        f.seek(0)
        f.truncate()
    orddir = open('data/Dset_all.txt')
    numpdb = 0
    list_0 = []
    list_1 = []
    list_02 = []
    list_24 = []
    list_46 = []
    list_68 = []
    list_80 = []
    for pdb in orddir:
        try:
            input_label = dict_pdb.get(pdb.strip())
            array = np.array(list(input_label))
            num0 = np.where(array == '0')[0].shape[0]
            num1 = np.where(array == '1')[0].shape[0]
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
    print(list_len)
    lenmin = np.min(list_len)
    random.shuffle(list_0)
    random.shuffle(list_1)
    random.shuffle(list_02)
    random.shuffle(list_24)
    random.shuffle(list_46)
    random.shuffle(list_68)
    random.shuffle(list_80)
    list_train = []
    with open('data/train.txt','a') as f:
        for i in range(1):
            list_train.append(list_1[i])
        for i in range(2):
            list_train.append(list_02[i])
        for i in range(16):
            list_train.append(list_24[i])
        for i in range(40):
            list_train.append(list_46[i])
        for i in range(150):
            list_train.append(list_68[i])
        for i in range(270):
            list_train.append(list_80[i])
        random.shuffle(list_train)
        for i in range(len(list_train)):
            f.write(list_train[i])
    list_eval = []
    with open('data/eval.txt','a') as f:
        for i in range(1):
            list_eval.append(list_1[i])
        for i in range(1):
            list_eval.append(list_02[i+2])
        for i in range(4):
            list_eval.append(list_24[i+16])
        for i in range(11):
            list_eval.append(list_46[i+40])
        for i in range(27):
            list_eval.append(list_68[i+150])
        for i in range(34):
            list_eval.append(list_80[i+270])
        random.shuffle(list_eval)
        for i in range(len(list_eval)):
            f.write(list_eval[i])



#生成非法数据集列表
'''
dict_seq, dict_pdb = getdict()
print(dict_pdb)
#Legitimacy_check(dict_pdb, dict_seq)
for pdb in open('data/eval.txt'):
    label = dict_seq.get(pdb.strip())
    print('>'+pdb.strip())
    print(label)
'''
