import os.path
import paddle
from MyNet import MyNet
import numpy as np
import math
def getevalute(label, site):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    lab = np.array(list(label))
    sit = np.array(list(site))
    for i in range(len(lab)):
        if lab[i] == sit[i]:
            if lab[i] == '0':
                TN += 1
            else:
                TP += 1
        else:
            if sit[i] == '0':
                FN += 1
            else:
                FP += 1
    return TP, TN, FP, FN

def compute_evalute(TP, TN, FP, FN):
    accu = float(TP+TN)/float(TP+TN+FP+FN)
    try:
        prec = float(TP)/float(TP+FP)
    except:
        prec = 0
    try:
        recall = float(TP)/float(TP+FN)
    except:
        recall = 0
    try:
        F1 = (2*prec*recall)/(prec+recall)
    except:
        F1 = 0
    try:
        MCC = float(TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    except:
        if FP == 0 and FN == 0:
            MCC = 1
        else:
            MCC = 0
    return round(accu,3),round(prec,3),round(recall,3),round(F1,3),round(MCC,3)


class runNet():

    def __init__(self,train_parameters,loader,length, index=None):
        super(runNet, self).__init__()
        self.train_parameters = train_parameters
        self.loader = loader
        self.length = length
        self.index = index

    def save(self):
        model = MyNet(self.length)
        model__state_dict = paddle.load('work/checkpoints/bssppi_save'+str(1+2*self.length)+'.pdparams')
        model.set_state_dict(model__state_dict)
        save_path = 'work/checkpoints/bssppi_save'+str(1+2*self.length)+'.pdparams.best'
        paddle.save(model.state_dict(), save_path)

    def train(self,pdb_id, len0, len1, mcc1):
        model = MyNet(self.length)
        if os.path.exists('work/checkpoints/bssppi_save'+str(1+2*self.length)+'.pdparams'):
            model__state_dict = paddle.load('work/checkpoints/bssppi_save'+str(1+2*self.length)+'.pdparams')
            model.set_state_dict(model__state_dict)
        model.train()
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        cross_entropy = paddle.nn.CrossEntropyLoss(weight=paddle.to_tensor(np.array([float((len0+len1)/len0),float((len0+len1)/len1)],dtype='float32')))
        optimizer = paddle.optimizer.SGD(learning_rate=self.train_parameters['learning_strategy']['lr'],
                                          parameters=model.parameters(),grad_clip = clip)

        steps = 0
        Iters, total_loss, total_acc, total_mcc = [], [], [], []
        issave = 0
        epo = 0
        tmpacc = len0 / (len0 + len1)
        posi = 0
        nega = 0
        while(issave < 20):
            if epo > 100:
                break
            for _, data in enumerate(self.loader()):
                label = ''
                site = ''
                y_true = []
                steps += 1
                x_data = data[0]
                tmplabel = data[1]
                y_data = paddle.to_tensor(np.array(data[1]).T,dtype='int64')
                index_data = data[2]
                predicts, acc= model([x_data, index_data], y_data)
                label = ''
                site = ''
                for i in range(len(predicts)):
                    if tmplabel[0][i]=='1':
                        label = label+'1'
                    else:
                        label = label+'0'
                    if predicts[i][0] > predicts[i][1]:
                        site = site+'0'
                    else:
                        site = site+'1'

                TP1, TN1, FP1, FN1 = getevalute(label, site)
                if posi == 0:
                    posi = TP1 + FN1
                    nega = TN1 + FP1
                _, precision, recall, f1, mcc = compute_evalute(TP1,TN1,FP1,FN1)
                loss = cross_entropy(predicts, y_data)
                loss.backward()
                optimizer.minimize(loss)
                optimizer.clear_grad()
                if steps % self.train_parameters["skip_steps"] == 0:
                    Iters.append(steps)
                    total_loss.append(loss.numpy())
                    total_acc.append(acc.numpy())
                    total_mcc.append(mcc)

                    if acc.numpy() >= tmpacc:
                        tmpacc = acc.numpy()
                        issave += 1
                        save_path = 'work/checkpoints/bssppi_save'+str(1+2*self.length)+'.pdparams.tmp'
                        paddle.save(model.state_dict(), save_path)
                        #print(TP1,TN1,FP1,FN1,precision,recall,f1,mcc)
                        #print('{}:id:{}, epo: {}, step: {}, loss is: {}, acc is: {}'.format(self.index, str(pdb_id), epo, steps, loss.numpy(), acc.numpy()))
                        #print('save model to: ' + save_path)

            epo += 1
        fp = open("train"+str(1+2*self.length)+".txt","a+",encoding="utf-8")
        fp.write(str(pdb_id)+' '+str(round(np.mean(total_loss), 3))+' '+str(round(np.mean(total_acc),3))+' '+str(round(np.mean(total_mcc),3))+' '+str(round(float(nega/(nega+posi)),3))+'\n')
        fp.close()
        if np.mean(total_mcc) >= mcc1:
            save_path = 'work/checkpoints/bssppi_save'+str(1+2*self.length)+'.pdparams'
            paddle.save(model.state_dict(), save_path)
            print(TP1,TN1,FP1,FN1,precision,recall,f1,mcc)
            print('{}:id:{}, epo: {}, step: {}, loss is: {}, acc is: {}'.format(self.index, str(pdb_id), epo, steps, loss.numpy(), acc.numpy()))
            print('save model to: ' + save_path)
            return np.mean(total_mcc)
        else:
            return mcc1


    def evalue(self,pdb_eval_id, label, site):
        '''
        模型评估
        '''
        model__state_dict = paddle.load('work/checkpoints/bssppi_save'+str(1+2*self.length)+'.pdparams')
        model_eval = MyNet(self.length)
        model_eval.set_state_dict(model__state_dict)
        model_eval.eval()
        for _, data in enumerate(self.loader()):
            x_data = data[0]
            tmplabel = data[1]
            y_data = paddle.to_tensor(np.array(data[1]).T,dtype='int64')
            index_data = data[2]
            predicts = model_eval([x_data, index_data])
            for i in range(len(predicts)):
                if tmplabel[0][i]=='1':
                    label = label+'1'
                else:
                    label = label+'0'
                if predicts[i][0] > predicts[i][1]:
                    site = site+'0'
                else:
                    site = site+'1'

        return label, site
