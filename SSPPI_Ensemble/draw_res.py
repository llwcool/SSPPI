import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

def draw_epoch_res(train_res, eval_res, resdir, epoch=100):
    fig_name = ["loss", "auroc", "auprc", "accuracy", "precision", "recall", "f1-score", "MCC"]
    label = [i for i in range(1, epoch + 1)]
    for i in range(len(fig_name)):
        t_res = [float(row[i]) for row in train_res]
        e_res = [float(row[i]) for row in eval_res]
        max_res = round(float(max(max(t_res), max(e_res))), 1)
        min_res = round(float(min(min(t_res), min(e_res))), 1)
        plt.figure()
        plt.plot(label, t_res, label="train", color = "steelblue")
        plt.plot(label, e_res, label = "eval", color = "skyblue")
        plt.ylim(min_res - 0.1, max_res + 0.1)
        plt.yticks(np.arange(min_res - 0.1, max_res + 0.1, 0.1))
        plt.xlabel("Epochs")
        plt.ylabel("Scores")
        plt.title(fig_name[i])
        plt.legend(loc = "lower right")
        plt.savefig(f"{resdir}/{fig_name[i]}.tif")
        plt.close()

def draw_loss(train_loss, eval_loss, resdir):
    label = [i for i in range(1, len(train_loss) + 1)]
    plt.figure()
    plt.plot(label, train_loss, label="train", color = "steelblue")
    plt.plot(label, eval_loss, label = "eval", color = "skyblue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend(loc = "upper right")
    plt.savefig(f"{resdir}/loss.tif")
    plt.close()

def my_roc_curve(y_pred, y_label, figure_file):
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])
    lw = 1
    plt.plot(fpr[0], tpr[0],
         lw=lw, label= 'ROC curve (area = %0.3f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    plt.legend(loc="lower right")
    plt.savefig(figure_file)
    plt.close()
    return

def my_prc_curve(y_pred, y_label, figure_file):
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)
    plt.plot(lr_recall, lr_precision, lw = 1, label= ' (area = %0.3f)' % average_precision_score(y_label, y_pred))
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.savefig(figure_file)
    plt.close()
    return


