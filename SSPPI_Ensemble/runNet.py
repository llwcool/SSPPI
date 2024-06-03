import paddle
from MyNet import MyNet
import numpy as np
import math
import time
from parameters import getLoss

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
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    except:
        pass
    return int(TP), int(FP), int(TN), int(FN), round(sensitivity, 3), round(specificity, 3), round(recall, 3), round(
        precision, 3), round(MCC, 3), round(F1, 3), round(accuracy, 3)


class runNet():
    def __init__(self, loader, length):
        super(runNet, self).__init__()
        self.loader = loader
        self.length = length

    def train(self, epo, state_dict, f_record_out):
        start = time.time()
        model = MyNet()
        if state_dict != None:
            model.set_state_dict(state_dict)
        model.train()
        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")
        optimizer = paddle.optimizer.AdamW(
            learning_rate=0.001,
            parameters=model.parameters(),
            beta1=beta1,
            beta2=beta2,
            weight_decay=0.0001
        )
        steps = 0
        y_true_sum = []
        predicts_sum = []
        loss_sum = []
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        for _, data in enumerate(self.loader()):
            steps += 1
            x_data = data[0]
            tmplabel = data[1]
            predicts = model(paddle.to_tensor(x_data, dtype='float32'))
            y_data = paddle.to_tensor(np.array(data[1]), dtype='float32')
            one = paddle.to_tensor([1.], dtype='float32')
            fg_label = paddle.greater_equal(y_data, one)
            fg_num = paddle.sum(paddle.cast(fg_label, dtype='float32')) if paddle.sum(paddle.cast(
                fg_label, dtype='float32')) != 0 else paddle.to_tensor(len(y_data), dtype='float32')
            loss = getLoss(predicts, y_data, fg_num)
            for i in range(len(predicts)):
                predicts_sum.append(1/(1+np.exp(-float(predicts[i]))))
                y_true_sum.append(tmplabel[i])
            loss_sum.append(loss.numpy())
            loss.backward()
            optimizer.step()
            # for par in model.parameters():
            # print(par.shape)
            # if par.shape == [1024, 32]:
            #    print(par._grad_ivar())
            optimizer.clear_grad()
            if steps % 10 == 0:
                y_true_1 = np.array(y_true_sum)
                y_pred_1 = np.array(predicts_sum)
                fpr, tpr, thre = roc_curve(y_true_1, y_pred_1)
                sorted_pred = np.sort(y_pred_1)
                sorted_pred_descending = np.flip(
                    sorted_pred)  # from big to small
                num_of_1 = np.count_nonzero(y_true_1)
                threshold = round(sorted_pred_descending.item(num_of_1 - 1), 3)
                pred_binary = np.where(y_pred_1 >= threshold, 1, 0)
                TP, FP, TN, FN, sensitivity, specificity, recall, precision, MCC, F1_score, accuracy = CalculateEvaluationMetrics(
                    y_true_1, pred_binary)
                pre, rec, _ = precision_recall_curve(y_true_1, y_pred_1)
                auroc = auc(fpr, tpr)
                auprc = auc(rec, pre)
                print(f"epo:{epo}, steps:{steps}, loss: {np.mean(loss_sum):.3f}, AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}, Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1:{F1_score}, MCC:{MCC}, best_threshold:{threshold}")
                f_record_out.write(f"epo:{epo}, steps:{steps}, loss: {np.mean(loss_sum):.3f}, AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}, Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1:{F1_score}, MCC:{MCC}, best_threshold:{threshold}\n")
                f_record_out.flush()
        end = time.time()
        print(f"epo:{epo} spend time: {end-start}")
        f_record_out.write(f"epo:{epo} spend time: {end-start}\n")
        f_record_out.flush()
        return model.state_dict(), [round(np.mean(loss_sum), 3), round(auroc, 3), round(auprc, 3), round(accuracy, 3), round(precision, 3), round(recall, 3), round(F1_score, 3), round(MCC, 3)]
