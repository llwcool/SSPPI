import os
import random
import paddle
import sys
from dataset import dataset
from runNet import runNet
from generate_matrix import getdict
import numpy as np
import math
from MyNet import MyNet
import time

dict_seq, dict_pdb = getdict()
dict_eval = {}
L = int(sys.argv[1]) if sys.argv[1] != None else 1
gpu = sys.argv[2] if sys.argv[2] != None else 0
batch_size = int(sys.argv[3]) if sys.argv[3] != None else 256
paddle.device.set_device("gpu:"+gpu)
model = paddle.Model(MyNet(L))
model.summary((batch_size,L * 2  + 1,122))
train_sum = []
eval_sum = []
with open(os.path.join('data', "train.txt"), "r", encoding="utf-8") as f:
    info_train = f.readlines()

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
    print("TP: ", TP)
    print("FP: ", FP)
    print("TN: ", TN)
    print("FN: ", FN)
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

class runNet_eval():
    def __init__(self,loader,length, index=None):
        super(runNet_eval, self).__init__()
        self.loader = loader
        self.length = length
        self.index = index

    def evalue_auc(self, y_true, y_pred, model_state_dict):
        '''
        模型评估
        '''
        model_eval = MyNet(self.length)
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
            loss_tmp.append(loss.numpy()[0])
            for i in range(len(predicts)):
                y_pred.append(1/(1+np.exp(-float(predicts[i]))))
                y_true.append(tmplabel[i])
        return y_true, y_pred, np.mean(loss_tmp)

for i in range(len(info_train)):
    pdb_train_id = info_train[i].strip()
    train_sum.append(pdb_train_id)
for i in range(len(info_eval)):
    pdb_eval_id = info_eval[i].strip()
    eval_sum.append(pdb_eval_id)
train_dataset = dataset(train_sum, L, dict_seq, dict_pdb)
eval_dataset = dataset(eval_sum, L, dict_seq, dict_pdb, False)
def mytrain(train_dataset, j, state_dict_now):
    train_loader = paddle.io.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    myNet = runNet(train_loader, L)
    state_dict_pre = state_dict_now
    state_dict_now = myNet.train(j, state_dict_now)
    myNet.save(j, state_dict_now)
    y_true = []
    y_pred = []
    eval_loader = paddle.io.DataLoader(eval_dataset, batch_size= batch_size, shuffle=False)
    myNet_1 = runNet_eval(eval_loader, L)
    y_true, y_pred, loss = myNet_1.evalue_auc(y_true, y_pred, state_dict_now)

    from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix, precision_score, recall_score,  roc_auc_score, f1_score, precision_recall_curve, matthews_corrcoef
    y_true_1 = np.array(y_true)
    y_pred_1 = np.array(y_pred)
    fpr, tpr,thre = roc_curve(y_true_1, y_pred_1)
    sorted_pred = np.sort(y_pred_1)
    sorted_pred_descending = np.flip(sorted_pred)  # from big to small
    num_of_1 = np.count_nonzero(y_true_1)
    threshold = round(sorted_pred_descending.item(num_of_1 - 1), 3)
    pred_binary = np.where(y_pred_1 >= threshold, 1, 0)
    TP, FP, TN, FN, sensitivity, specificity, rec, pre, MCC, F1_score, accuracy = CalculateEvaluationMetrics(y_true_1, pred_binary)
    precision, recall, _ = precision_recall_curve(y_true_1, y_pred_1)
    print("loss: {}".format(loss))
    print("AUROC: {}".format(auc(fpr,tpr)))
    print("AUPRC: {}".format(auc(recall, precision)))
    print("Accuracy:{}".format(accuracy))
    print("Precision:{}".format(pre))
    print("Recall:{}".format(rec))
    print("F1:{}".format(F1_score))
    print("MCC:{}".format(MCC))
    print("best_threshold:{}".format(threshold))
    end = time.time()
    print(end - start)
    f1 = open('evaluating/result_'+str(L), 'a+')
    strofans = ("{}, AUROC:{}, AUPRC:{}, acc:{}, precision:{}, recall:{}, f1:{}, mcc:{}, loss:{}, threshold:{}".format(j, auc(fpr,tpr), auc(recall, precision), accuracy, pre, rec, F1_score, MCC, loss, threshold))
    f1.write(strofans+'\n')
    return auc(fpr,tpr), state_dict_now

best_path = 'work/checkpoints/bssppi_save'+str(1+2*L)+'_best'
state_dict= None
auroc_best = 0
for j in range(100):
    start = time.time()
    auroc, state_dict = mytrain(train_dataset, str(j), state_dict)
    if auroc > auroc_best:
        paddle.save(state_dict, best_path+'.pdparams')
        auroc_best = auroc
