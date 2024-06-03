import os
import paddle
from dataset import dataset
from dataset_SSPPI import dataset as dataset_SSPPI
from generate_matrix import getdict
import numpy as np
from MyNet import MyNet
from MyNet_onlyseq import MyNet as MyNet_onlyseq
from MyNet_onlystr import MyNet as MyNet_onlystr
from MyNet_all import MyNet as MyNet_all
import sys
import math
from sklearn.metrics import roc_curve, auc, precision_recall_curve, matthews_corrcoef

distance = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
length = [5, 8, 12, 15, 20, 25, 32, 38, 46, 53, 60, 69]
dict_length_distance = dict(zip(distance, length))

dict_seq, dict_pdb = getdict()
dict_eval = {}
path_0 = sys.argv[1] # ensemble
path_1 = sys.argv[2] # all
path_2 = sys.argv[3] # only_seq
path_3 = sys.argv[4] # only_str

data = "data/eval.txt"
# 数据加载
accs = []
precisions = []
recalls = []
f1s = []
mccs = []
with open(os.path.join('data', "eval.txt"), "r", encoding="utf-8") as f:
    info_eval = f.readlines()

def CalculateEvaluationMetrics(y_true, y_pred):
    TP = float(0)
    FP = float(0)
    TN = float(0)
    FN = float(0)
    sensitivity = 0
    specificity = 0
    recall = 0
    precision = 0
    MCC = 0
    F1 =0 
    accuracy = 0
    
    for i, j in zip(y_true, y_pred):
        if (i == 1 and j == 1):
            TP += 1
        elif (i == 0 and j == 1):
            FP += 1
        elif (i == 0 and j == 0):
            TN += 1
        elif (i == 1 and j == 0):
            FN += 1
    try:
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        F1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
    except:
        pass
    return int(TP), int(FP), int(TN), int(FN), round(sensitivity, 3), round(specificity, 3), round(recall,3), round(precision,3), round(MCC,3), round(F1,3), round(accuracy,3)

class runNet_SSPPI():
    def __init__(self,loader,length, index=None):
        super(runNet_SSPPI, self).__init__()
        self.loader = loader
        self.length = length
        self.index = index

    def evalue_auc(self, y_true, y_pred, model_state_dict, category = None):
        '''
        模型评估
        '''
        if category == None:
            model_eval = MyNet_all(self.length, True)
        elif category == "onlyseq":
            model_eval = MyNet_onlyseq(self.length, True)
        elif category == "onlystr":
            model_eval = MyNet_onlystr(self.length, True)
        model_eval.set_state_dict(model_state_dict)
        model_eval.eval()
        for _, data in enumerate(self.loader()):
            x_data = data[0]
            tmplabel = data[1]
            predicts= model_eval(paddle.to_tensor(x_data, dtype='float32'))
            for i in range(len(predicts)):
                y_pred.append(np.array(predicts[i]))
                y_true.append(np.array(tmplabel[i]))
        return y_true, y_pred

class runNet_ensemble():
    def __init__(self,loader,length, index=None):
        super(runNet_ensemble, self).__init__()
        self.loader = loader
        self.length = length
        self.index = index

    def evalue_auc(self, y_true, y_pred, model_state_dict):
        '''
        模型评估
        '''
        model_eval = MyNet()
        model_eval.set_state_dict(model_state_dict)
        model_eval.eval()
        loss_tmp = [] 
        for _, data in enumerate(self.loader()):
            x_data = data[0]
            tmplabel = data[1]
            y_data = paddle.to_tensor(np.array(data[1]),dtype='float32')
            predicts= model_eval(paddle.to_tensor(x_data, dtype='float32'))
            one = paddle.to_tensor([1.], dtype='float32')
            fg_label = paddle.greater_equal(y_data, one)
            fg_num = paddle.sum(paddle.cast(fg_label, dtype='float32')) if paddle.sum(paddle.cast(fg_label, dtype='float32'))!=0 else paddle.to_tensor(len(y_data), dtype='float32')
            loss = paddle.nn.functional.sigmoid_focal_loss(predicts, y_data, normalizer=fg_num, alpha = 0.85)
            loss_tmp.append(loss.numpy())
            for i in range(len(predicts)):
                y_pred.append(1/(1+np.exp(-float(predicts[i]))))
                y_true.append(tmplabel[i])
        return y_true, y_pred, np.mean(loss_tmp)
gpu = "0"
if gpu != "cpu": 
    paddle.device.set_device("gpu:"+gpu)
else:
    paddle.device.set_device("cpu")
#state_dict= None
state_dict_0= paddle.load(path_0)
state_dict_1= paddle.load(path_1)
state_dict_2= paddle.load(path_2)
state_dict_3= paddle.load(path_3)
train_sum = []
eval_sum = []
y_true_sum = []
y_pred_sum = []
f = open(f"result_ensemble.txt", "a+")
f.write(f"protein\t TP\t FP\t TN\t FN\t AUROC\t AUPRC\t Accuracy\t Precision\t Recall\t F1\t MCC\t \n")
dict_pdb_score = {}
for i in range(len(info_eval)):
    pdb_eval_id = info_eval[i].strip()
    input_label = dict_pdb.get(pdb_eval_id)

    # eval_sum.append(pdb_eval_id)
    eval_dataset_all = dataset_SSPPI([pdb_eval_id], 1, dict_seq, dict_pdb, 7, dict_length_distance[7], False)
    eval_loader_all = paddle.io.DataLoader(eval_dataset_all, batch_size= len(input_label), shuffle=False)
    myNet_all = runNet_SSPPI(eval_loader_all, 1)
    _, y_pred_1= myNet_all.evalue_auc([], [], state_dict_1)
    np.save(f"ensemble_data/{pdb_eval_id}_pred_all.npy", y_pred_1)

    eval_dataset_onlyseq = dataset_SSPPI([pdb_eval_id], 15, dict_seq, dict_pdb, 4, dict_length_distance[4], True)
    eval_loader_onlyseq = paddle.io.DataLoader(eval_dataset_onlyseq, batch_size= len(input_label), shuffle=False)
    myNet_onlyseq = runNet_SSPPI(eval_loader_onlyseq, 15)
    _, y_pred_2 = myNet_onlyseq.evalue_auc([], [], state_dict_2, "onlyseq")
    np.save(f"ensemble_data/{pdb_eval_id}_pred_onlyseq.npy", y_pred_2)

    eval_dataset_onlystr = dataset_SSPPI([pdb_eval_id], 1, dict_seq, dict_pdb, 14, dict_length_distance[14], False)
    eval_loader_onlystr = paddle.io.DataLoader(eval_dataset_onlystr, batch_size= len(input_label), shuffle=False)
    myNet_onlystr = runNet_SSPPI(eval_loader_onlystr, 1)
    _, y_pred_3= myNet_onlystr.evalue_auc([], [], state_dict_3, "onlystr")
    np.save(f"ensemble_data/{pdb_eval_id}_pred_onlystr.npy", y_pred_3)

    eval_dataset = dataset([pdb_eval_id], dict_seq, dict_pdb)
    eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=len(input_label), shuffle=False)
    myNet = runNet_ensemble(eval_loader, 1)
    y_true, y_pred, _ = myNet.evalue_auc([], [], state_dict_0)

    y_true_sum.extend(y_true)
    y_pred_sum.extend(y_pred)
    dict_pdb_score[pdb_eval_id] = y_pred
y_true = np.array(y_true_sum)
y_pred = np.array(y_pred_sum)
np.save("y_true.npy", y_true)
np.save("y_pred_ensemble.npy", y_pred)
fpr, tpr, thre = roc_curve(y_true, y_pred)
thresholds = np.arange(0, 1.01, 0.01)
mcc_list = [matthews_corrcoef(y_true, y_pred > t) for t in thresholds]
threshold = thresholds[np.argmax(mcc_list)]
pred_binary = np.where(y_pred >= threshold, 1, 0)
TP, FP, TN, FN, sensitivity, specificity, rec, pre, MCC, F1_score, accuracy = CalculateEvaluationMetrics(y_true, pred_binary)
precision, recall, _ = precision_recall_curve(y_true, y_pred)
print(f"All protein, {TP}, {FP}, {TN}, {FN}, AUROC: {auc(fpr,tpr)}, AUPRC: {auc(recall, precision)}, Accuracy:{accuracy}, Precision:{pre}, Recall:{rec}, F1:{F1_score}, MCC:{MCC}, best_threshold:{threshold}")
f.write(f"all\t {TP}\t {FP}\t {TN}\t {FN}\t {auc(fpr,tpr)}\t {auc(recall, precision)}\t {accuracy}\t {pre}\t {rec}\t {F1_score}\t {MCC}\t {threshold}\n")

for pdb in dict_pdb_score:
    label = dict_pdb.get(pdb)
    y_true = [int(i) for i in str(label)]
    y_pred = dict_pdb_score.get(pdb)
    fpr, tpr,thre = roc_curve(y_true, y_pred)
    pred_binary = np.where(y_pred >= threshold, 1, 0)
    TP, FP, TN, FN, sensitivity, specificity, rec, pre, MCC, F1_score, accuracy = CalculateEvaluationMetrics(y_true, pred_binary)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    f.write("%s\t %d\t %d\t %d\t %d\t %f\t %f\t %f\t %f\t %f\t %f\t %f\n" % (pdb, TP, FP, TN, FN, auc(fpr,tpr), auc(recall, precision), accuracy, pre, rec, F1_score, MCC))
