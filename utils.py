import os
import pprint
import torch
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

all_true_labels = []  # 存储所有任务的真实标签
all_pred_labels = []  # 存储所有任务的预测标签




def draw_confusion_matrix1(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.figure(figsize=(25, 25))
    plt.imshow(cm, cmap='Blues')
    # plt.title(title,fontsize=40)
    # plt.xlabel("Predict label",fontsize=30)
    # plt.ylabel("Truth label",fontsize=30)
    plt.yticks(range(label_name.__len__()), label_name,fontsize=35,fontweight='bold')
    plt.xticks(range(label_name.__len__()), label_name, fontsize=35,fontweight='bold')

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    plt.tight_layout()

    # plt.colorbar()
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=30)

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            if i == j:
                plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color,
                         fontsize=36,fontweight='bold')
            else:
                plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color,
                         fontsize=36)

    plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    # plt.title(title)
    # plt.xlabel("Predict label")
    # plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name,fontsize=15.5)
    plt.xticks(range(label_name.__len__()), label_name,fontsize=15.5)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()

    # plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color,fontsize=15.5)

    plt.show()
    # if not pdf_save_path is None:
    #     plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)



def ensure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    # print(pred.size())

    pred1 = pred.cpu().detach().numpy()
    label1 = label.cpu().detach().numpy()
    # draw_confusion_matrix(label_true=label1,  # y_gt=[0,5,1,6,3,...]
    #                       label_pred=pred1,  # y_pred=[0,5,1,6,3,...]
    #                       label_name=["0", "1", "2", "3", "4"],
    #                       title="Confusion Matrix on Fer2013",
    #                       pdf_save_path="Confusion_Matrix_on_Fer2013.jpg",
    #                       dpi=300)


    # if i > 1:
    #     all_true_labels.extend(pred1)
    #     all_pred_labels.extend(label1)

    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item(), pred1, label1
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def centering(support, query):
    # support: # *, c
    # query: *, c
    support = support - support.mean(dim=-1, keepdim=True)
    query = query - query.mean(dim=-1, keepdim=True)
    return support, query

def _l2norm(x, dim=1, keepdim=True):
    return x / (1e-16 + torch.norm(x, 2, dim, keepdim))
