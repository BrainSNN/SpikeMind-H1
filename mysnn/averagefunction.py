import torch
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def shuffle(x, y):
    x = np.array(x)
    y = np.array(y)
    index = [i for i in range(len(x))]
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    # x=x.tolist()
    # y=y.tolist()
    return x, y

def evaluate_accuracys(img, label, net, batch_size=16, B=True,per=(0,1,2,3)):
    loop_times = round(len(img) / batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_sum, n = 0.0, 0
    RES = []
    net.eval()
    for i in tqdm(range(1, loop_times)):
        data = torch.Tensor(img[(i - 1) * batch_size:i * batch_size]).float()
        lab = torch.Tensor(label[(i - 1) * batch_size:i * batch_size])
        # data=data.view(batch_size,1,64,500)
        with torch.no_grad():
            data = torch.unsqueeze(data, 1)  # 对齐维度
            data = data.permute(per)

            if isinstance(net, torch.nn.Module):
                outputs = net(data.to(device))[0]
                OUT = outputs.argmax(dim=1)
                acc_sum += (OUT == lab.to(device)).float().sum().cpu().item()
                RES.append(OUT)
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(data, is_training=False).argmax(dim=1) == lab).float().sum().item()
                else:
                    acc_sum += (net(data).argmax(dim=1) == lab).float().sum().item()
            n += lab.shape[0]
    if B:
        pre = torch.cat(RES, dim=0)
        real = torch.tensor(label[:len(pre)])
        pre = pre.cpu().tolist()
        real = real.tolist()
        get_classification_scores(real, pre)
    net.train()
    return acc_sum / n

def old_evaluate_accuracys(img, label, net, batch_size=16, B=True):
    loop_times = round(len(img) / batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_sum, n = 0.0, 0
    RES = []
    for i in tqdm(range(1, loop_times+1)):
        data = torch.Tensor(img[(i - 1) * batch_size:i * batch_size])
        lab = torch.Tensor(label[(i - 1) * batch_size:i * batch_size])
        with torch.no_grad():
            data = torch.unsqueeze(data, 1)
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                outputs = net(data.to(device))
                OUT = outputs.argmax(dim=1)
                acc_sum += (OUT == lab.to(device)).float().sum().cpu().item()
                RES.append(OUT)
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(data, is_training=False).argmax(dim=1) == lab).float().sum().item()
                else:
                    acc_sum += (net(data).argmax(dim=1) == lab).float().sum().item()
            n += lab.shape[0]
    if B:
        pre = torch.cat(RES, dim=0)
        real = torch.tensor(label[:len(pre)])
        pre = pre.cpu().tolist()
        real = real.tolist()
        get_classification_scores(real, pre)

    return acc_sum / n

def get_classification_scores(gt, pred):
    [[tp, fp], [fn, tn]] = standard_confusion_matrix(gt, pred)
    msg = (f'  - Confusion Matrix:\n'
           '    -----------------------\n'
           f'    | TP: {tp:4.0f} | FP: {fp:4.0f} |\n'
           '    -----------------------\n'
           f'    | FN: {fn:4.0f} | TN: {tn:4.0f} |\n'
           f'    TP+FN: {tp + fn:4.0f} | FP+TN: {fp + tn:4.0f} | {len(gt)}\n'
           '    -----------------------')
    print(msg)
    # TPR(sensitivity), TNR(specificity)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    # Precision, Recall, F1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    msg = ('  - Classification:\n'
           '      TPR/Sensitivity: {0:6.4f}\n'
           '      TNR/Specificity: {1:6.4f}\n'
           '      Precision: {2:6.4f}\n'
           '      Recall: {3:6.4f}\n'
           '      F1-score: {4:6.4f}').format(tpr, tnr, precision, recall, f1_score)
    print(msg)

    return tpr, tnr, precision, recall, f1_score


def standard_confusion_matrix(gt, pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = metrics.confusion_matrix(np.asarray(gt), np.asarray(pred))
    return np.array([[tp, fp], [fn, tn]])