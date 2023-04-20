import numpy as np
from paddle.io import Dataset
from generate_matrix import generatematrix

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

class dataset(Dataset):
    def __init__(self, pdb_id_sum, length, dict_seq, dict_pdb, is_train = True):
        super().__init__()
        self.pdb_id_sum = pdb_id_sum
        self.length = length
        self.input_matrix_sum = []
        self.labels = []
        self.dict_seq = dict_seq
        self.dict_pdb = dict_pdb
        self.is_train = is_train
        #20, 20, 20,20, 14, 29
        for pdb_id in pdb_id_sum:
            input_matrix, input_label, _ = generatematrix(pdb_id, self.dict_pdb, self.dict_seq)
            self.lab = np.array(list(input_label))
            np_expansion_1 = input_matrix[:self.length, :]
            np_expansion_2 = input_matrix[-self.length:, :]
            input_matrix = np.concatenate((np.concatenate((np_expansion_1, input_matrix), axis=0), np_expansion_2), axis=0).astype(float)
            input_matrix[:, :20] = normalization(input_matrix[:,:20])
            input_matrix[:, 20:40] = normalization(input_matrix[:,20:40])
            input_matrix[:, 94:] = normalization(input_matrix[:,94:])
            for i in range(len(input_label)):
                self.input_matrix_sum.append(input_matrix[i:i + 2 * self.length + 1, :])
                self.labels.append([int(input_label[i])])
        self.input_matrix_sum = np.array(self.input_matrix_sum)
        self.labels = np.array(self.labels)
        
    def __getitem__(self, index):
        return self.input_matrix_sum[index], self.labels[index]
        
    def __len__(self):
        return len(self.labels)
