import numpy as np
from paddle.io import Dataset
from generate_matrix import generatematrix
import os

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

class dataset(Dataset):
    def __init__(self, pdb_id_sum, length, dict_seq, dict_pdb, dis, win_length, is_train = True):
        super().__init__()
        self.pdb_id_sum = pdb_id_sum
        self.length = length
        self.input_matrix_seq = []
        self.labels = []
        self.dict_seq = dict_seq
        self.dict_pdb = dict_pdb
        self.is_train = is_train
        #20, 20, 20,20, 14, 29
        for pdb_id in pdb_id_sum:
            dis_map = np.load("features/distance_map/" + pdb_id.replace("_", "") + ".npy")
            if os.path.exists("features/matrix/" + pdb_id + "_input.npy") and os.path.exists("features/matrix/" + pdb_id + "_label.npy"):
                input_matrix = np.load("features/matrix/" + pdb_id + "_input.npy")
                input_label = np.load("features/matrix/" + pdb_id + "_label.npy")
            else:
                input_matrix, input_label = generatematrix(pdb_id, self.dict_pdb, self.dict_seq)
            input_label = [int(i) for i in str(input_label)]

            input_matrix[:, :20] = normalization(input_matrix[:,:20])
            input_matrix[:, 20:40] = normalization(input_matrix[:,20:40])
            input_matrix[:, 80:94] = normalization(input_matrix[:,80:94])
            input_matrix[:, 94:103] = normalization(input_matrix[:,94:103])
            input_matrix[:, 103:113] = normalization(input_matrix[:,103:113])
            input_matrix[:, 113:121] = normalization(input_matrix[:,113:121])
            
            np_expansion_1 = np.zeros(input_matrix[:self.length, :].shape)
            np_expansion_2 = np.zeros(input_matrix[-self.length:, :].shape)
            input_matrix = np.concatenate((np.concatenate((np_expansion_1, input_matrix), axis=0), np_expansion_2), axis=0).astype(float)
            

            for i in range(len(input_label)):
                tmp_feature = input_matrix[i:i + 2 * self.length + 1, :]
                self.labels.append([int(input_label[i])])

                tmp_dict_map = {}
                for j in range(len(dis_map[i])):
                    tmp_dict_map.update({dis_map[j][i] : j})
                tmp_struct_feature = []
                for j in sorted(tmp_dict_map):
                    tmp_struct_feature.append(input_matrix[tmp_dict_map[j], :])
                    if j > dis:
                        break
                tmp_struct_feature = np.array(tmp_struct_feature)
                if len(tmp_struct_feature) < win_length:
                    tmp_struct_feature = np.concatenate((tmp_struct_feature, np.zeros((win_length-len(tmp_struct_feature), 122))), axis=0)
                if len(tmp_struct_feature) > win_length:
                    tmp_struct_feature = tmp_struct_feature[:win_length,:]
                self.input_matrix_seq.append(np.concatenate((tmp_feature, tmp_struct_feature), axis = 0))
                
        self.input_matrix_seq = np.array(self.input_matrix_seq)
        self.labels = np.array(self.labels)
        
    def __getitem__(self, index):
        return self.input_matrix_seq[index], self.labels[index]
        
    def __len__(self):
        return len(self.labels)
