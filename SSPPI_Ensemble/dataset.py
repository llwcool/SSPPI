import numpy as np
import paddle
from paddle.io import Dataset
from generate_matrix import generatematrix, getdict
import os

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

class dataset(Dataset):
    def __init__(self, pdb_id_sum, dict_seq, dict_pdb):
        super().__init__()
        self.pdb_id_sum = pdb_id_sum
        self.input_matrix = []
        self.labels = []
        self.dict_seq = dict_seq
        self.dict_pdb = dict_pdb
        #20, 20, 20,20, 14, 29
        for pdb_id in pdb_id_sum:
            pdb_id = pdb_id.strip()
            input_feature, _= generatematrix(pdb_id, self.dict_pdb, self.dict_seq)
            input_label = self.dict_pdb.get(pdb_id)
            input_label = [int(i) for i in str(input_label)]

            input_feature[:, :20] = normalization(input_feature[:,:20])
            input_feature[:, 20:40] = normalization(input_feature[:,20:40])
            input_feature[:, 80:94] = normalization(input_feature[:,80:94])
            input_feature[:, 94:103] = normalization(input_feature[:,94:103])
            input_feature[:, 103:113] = normalization(input_feature[:,103:113])
            input_feature[:, 113:121] = normalization(input_feature[:,113:121])

            pred_all = np.load(f"ensemble_data/{pdb_id}_pred_all.npy")
            pred_onlyseq = np.load(f"ensemble_data/{pdb_id}_pred_onlyseq.npy")
            pred_onlystr = np.load(f"ensemble_data/{pdb_id}_pred_onlystr.npy")
            input_matrix = np.concatenate((np.concatenate((pred_all, pred_onlyseq), axis=1), pred_onlystr), axis=1).astype(float)
            input_matrix = np.concatenate((input_matrix, input_feature), axis=1)
            for i in range(len(input_label)):
                self.labels.append([int(input_label[i])])
                tmp_matrix = np.append(input_matrix[i], len(input_label))
                self.input_matrix.append(tmp_matrix)
                
        self.input_matrix = np.array(self.input_matrix)
        self.labels = np.array(self.labels)
        
    def __getitem__(self, index):
        return self.input_matrix[index], self.labels[index]
        
    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    dict_seq, dict_pdb = getdict()
    with open(os.path.join('data', "eval.txt"), "r", encoding="utf-8") as f:
        info_eval = f.readlines()
    eval_dataset = dataset(info_eval, dict_seq, dict_pdb)
    eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=1, shuffle=False)
    for _, data in enumerate(eval_loader()):
        print(data)
        break
    print("done")
