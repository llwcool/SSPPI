import os
import paddle
from dataset import dataset
from runNet import runNet
import numpy as np
import math
from MyNet import MyNet
from draw_res import draw_epoch_res, my_roc_curve, my_prc_curve, draw_loss
import shutil
from parameters import getLoss

def start_train(dict_seq, dict_pdb, L, gpu, batch_size, info_train, info_eval, epo):
    batch_size = int(batch_size)
    paddle.device.set_device(f"gpu:{gpu}")
    train_sum = []
    eval_sum = []
    for i in range(len(info_train)):
        pdb_train_id = info_train[i].strip()
        train_sum.append(pdb_train_id)
    for i in range(len(info_eval)):
        pdb_eval_id = info_eval[i].strip()
        eval_sum.append(pdb_eval_id)
    train_dataset = dataset(train_sum, dict_seq, dict_pdb)
    eval_dataset = dataset(eval_sum, dict_seq, dict_pdb)
    resdir = f"work/result_{L}"
    if os.path.exists(resdir):
        shutil.rmtree(resdir)
    os.mkdir(resdir)
    best_path = resdir + '/best'
    state_dict = None
    auroc_best = 0
    true_best = []
    pred_best = []
    best_eval_res = []
    all_train_res = []
    all_eval_res = []
    train_loss, eval_loss = [], []
    min_loss = 100
    lossNochange = 0
    init_loss = 0
    retrain = 0
    j = 0
    while j < epo:
        # def mytrain(train_dataset, eval_dataset, j, state_dict_now, batch_size, L, dis, win_length, start):
        if retrain > 10 or lossNochange > 20:
            break
        auroc, state_dict_now, y_true, y_pred, train_res, eval_res = mytrain(train_dataset, eval_dataset, str(j), state_dict, batch_size, L, resdir)
        if j == 0:
            init_loss = eval_res[0]
            state_dict = state_dict_now
            retrain = 0
        else:
            if eval_res[0] < init_loss * 2:
                state_dict = state_dict_now
                retrain = 0
            else:
                retrain += 1
                continue
        
        if eval_res[0] < min_loss:
            min_loss = eval_res[0]
            lossNochange = 0
        else:
            lossNochange += 1
        
        if auroc >= auroc_best and eval_res[-1] > 0.15:
            paddle.save(state_dict, best_path + f"_{j}.pdparams")
            true_best = y_true
            pred_best = y_pred
            auroc_best = auroc
            best_eval_res = eval_res
        
        all_train_res.append(train_res)
        all_eval_res.append(eval_res)
        train_loss.append(train_res[0])
        eval_loss.append(eval_res[0])
        draw_loss(train_loss, eval_loss, resdir)
        j += 1

    with open(f'{resdir}/best_eval.result', 'w') as f:
        f.write(f'loss: {best_eval_res[0]}, auroc: {best_eval_res[1]}, auprc: {best_eval_res[2]}, accuracy: {best_eval_res[3]}, precision: {best_eval_res[4]}, recall: {best_eval_res[5]}, F1_score: {best_eval_res[6]}, MCC: {best_eval_res[7]}\n')
    file1 = open(f'{resdir}/best.result', "a+")
    for i in range(len(true_best)):
        file1.write(str(true_best[i][0]) + '\t' + str(pred_best[i]) + '\n')
    file1.flush()
    file1.close()
    draw_epoch_res(all_train_res, all_eval_res, resdir, len(all_train_res))
    my_roc_curve(pred_best, true_best, f"{resdir}/roc_curve.tif")
    my_prc_curve(pred_best, true_best, f"{resdir}/prc_curve.tif")

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
    F1 = 0
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
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP)
                                              * (TP + FN) * (TN + FP) * (TN + FN))
        F1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
    except:
        pass
    return int(TP), int(FP), int(TN), int(FN), round(sensitivity, 3), round(specificity, 3), round(recall, 3), round(precision, 3), round(MCC, 3), round(F1, 3), round(accuracy, 3)

class runNet_eval():
    def __init__(self, loader, length, index=None):
        super(runNet_eval, self).__init__()
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
            y_data = paddle.to_tensor(np.array(data[1]), dtype='float32')
            predicts = model_eval(paddle.to_tensor(x_data, dtype='float32'))
            one = paddle.to_tensor([1.], dtype='float32')
            fg_label = paddle.greater_equal(y_data, one)
            fg_num = paddle.sum(paddle.cast(fg_label, dtype='float32')) if paddle.sum(paddle.cast(
                fg_label, dtype='float32')) != 0 else paddle.to_tensor(len(y_data), dtype='float32')
            loss = getLoss(predicts, y_data, fg_num)
            loss_tmp.append(loss.numpy())
            for i in range(len(predicts)):
                y_pred.append(1/(1+np.exp(-float(predicts[i]))))
                y_true.append(tmplabel[i])
        return y_true, y_pred, np.mean(loss_tmp)

def mytrain(train_dataset, eval_dataset, j, state_dict_now, batch_size, L, resdir):
    train_loader = paddle.io.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    myNet = runNet(train_loader, L)
    f_record_out = open(f"{resdir}/record_train", "a+")
    # return model.state_dict(), round(np.mean(_loss), 3), round(auroc, 3), round(auprc, 3), round(accuracy, 3), round(precision, 3), round(recall, 3), round(F1_score, 3), round(MCC, 3)
    state_dict_now, train_res = myNet.train(j, state_dict_now, f_record_out)
    # myNet.save(j, state_dict_now)
    y_true = []
    y_pred = []
    eval_loader = paddle.io.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False)
    myNet_1 = runNet_eval(eval_loader, L)
    y_true, y_pred, loss = myNet_1.evalue_auc(y_true, y_pred, state_dict_now)

    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    y_true_1 = np.array(y_true)
    y_pred_1 = np.array(y_pred)
    fpr, tpr, thre = roc_curve(y_true_1, y_pred_1)
    sorted_pred = np.sort(y_pred_1)
    sorted_pred_descending = np.flip(sorted_pred)  # from big to small
    num_of_1 = np.count_nonzero(y_true_1)
    threshold = round(sorted_pred_descending.item(num_of_1 - 1), 3)
    pred_binary = np.where(y_pred_1 >= threshold, 1, 0)
    TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, F1_score, accuracy = CalculateEvaluationMetrics(
        y_true_1, pred_binary)
    pre, rec, _ = precision_recall_curve(y_true_1, y_pred_1)
    auroc = auc(fpr, tpr)
    auprc = auc(rec, pre)
    print(f'epoch:{j}, \n TP: {TP}, \t FP: {FP}, \n TN: {TN}, \t FN: {FN}, \n loss: {loss:.3f}, AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}, Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1:{F1_score}, MCC:{MCC}, best_threshold:{threshold}')
    f1 = open(f'{resdir}/record_eval', 'a+')
    f1.write(f'epoch:{j}, \n TP: {TP}, \t FP: {FP}, \n TN: {TN}, \t FN: {FN}, \n loss: {loss:.3f}, AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}, Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1:{F1_score}, MCC:{MCC}, best_threshold:{threshold}\n')
    f1.flush()
    eval_res = [round(loss, 3), round(auroc, 3), round(auprc, 3), round(accuracy, 3), round(precision, 3), round(recall, 3), round(F1_score, 3), round(MCC, 3)]
    return auc(fpr, tpr), state_dict_now, y_true_1, y_pred_1, train_res, eval_res
