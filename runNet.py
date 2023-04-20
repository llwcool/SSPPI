import os.path
import paddle
from MyNet import MyNet
import numpy as np
import math

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
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        F1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    except:
        pass
    return int(TP), int(FP), int(TN), int(FN), round(sensitivity, 3), round(specificity, 3), round(recall, 3), round(
        precision, 3), round(MCC, 3), round(F1, 3), round(accuracy, 3)


class runNet():

    def __init__(self,loader,length):
        super(runNet, self).__init__()
        self.loader = loader
        self.length = length
        self.learning_rate = 0.001
       
    def save(self, epo, state_dict):
        save_path = 'work/checkpoints/bssppi_save'+str(1+2*self.length)+'_'+epo
        paddle.save(state_dict, save_path+'.pdparams')

    def train(self, epo, state_dict):
        model = MyNet(self.length)
        if state_dict != None:
            model.set_state_dict(state_dict)
        model.train()
        optimizer = paddle.optimizer.AdamW(learning_rate= 0.001, parameters=model.parameters())
        steps = 0
        _loss= []
        _acc= []
        _mcc= []
        y_true_sum = []
        predicts_sum = []
        loss_sum = []
        from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix, precision_score, \
                    recall_score, roc_auc_score, f1_score, precision_recall_curve
        for _, data in enumerate(self.loader()):
            steps += 1
            x_data = data[0]
            tmplabel = data[1]
            predicts= model(paddle.to_tensor(x_data, dtype='float32'))
            y_data = paddle.to_tensor(np.array(data[1]), dtype='float32')
            one = paddle.to_tensor([1.], dtype='float32')
            fg_label = paddle.greater_equal(y_data, one)
            fg_num = paddle.sum(paddle.cast(fg_label, dtype='float32')) if paddle.sum(paddle.cast(fg_label, dtype='float32'))!=0 else paddle.to_tensor(len(y_data), dtype='float32')
            loss = paddle.nn.functional.sigmoid_focal_loss(predicts, y_data, normalizer=fg_num, alpha = 0.85)
            for i in range(len(predicts)):
                predicts_sum.append(1/(1+np.exp(-float(predicts[i]))))
                y_true_sum.append(tmplabel[i])
            
            _loss.append(loss.numpy()[0])
            loss_sum.append(loss.numpy()[0])
            loss.backward()
            optimizer.step()
            #for par in model.parameters():
                    #print(par.shape)
                    #if par.shape == [1024, 32]:
            #    print(par._grad_ivar())
            optimizer.clear_grad()
            if steps % 50 == 0:
                y_true_1 = np.array(y_true_sum)
                y_pred_1 = np.array(predicts_sum)
                fpr, tpr, thre = roc_curve(y_true_1, y_pred_1)
                sorted_pred = np.sort(y_pred_1)
                sorted_pred_descending = np.flip(sorted_pred)  # from big to small
                num_of_1 = np.count_nonzero(y_true_1)
                threshold = round(sorted_pred_descending.item(num_of_1 - 1), 3)
                pred_binary = np.where(y_pred_1 >= threshold, 1, 0)
                TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, F1_score, accuracy = CalculateEvaluationMetrics(y_true_1, pred_binary)
                print('epo:{}, steps:{}, loss is: {:.3f}, acc is: {:.3f}, mcc is: {:.3f}'.format(epo, steps, round(np.mean(loss_sum),3), round(accuracy, 3), round(MCC, 3)))
                _acc.append(accuracy)
                _mcc.append(MCC)
                y_true_sum = []
                predicts_sum = []
                loss_sum = []
        print("epo:{}, acc is :{}, loss is: {}, mcc is:{}".format(epo, np.mean(_acc), np.sum(_loss), np.mean(_mcc)))
        return model.state_dict()
