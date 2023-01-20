import numpy as np
import torch
import sys
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
import torch.nn.functional as F

#Given a model and a data_loader, produce the metrics of that data set
def run_test(model, test_loader):
    model.eval()    #turn on evaluation mode
    y_pred_lst = []
    y_true_lst = []
    probs_lst = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            if torch.cuda.is_available():
                x = x.cuda()
            _, logit = model(x)
            y_pred = torch.argmax(logit, dim = 1).tolist()
            y_true = y.tolist()
            y_softmax = F.softmax(logit, dim = 1)[:,1]
            probs_lst.extend(y_softmax.tolist())
            y_pred_lst.extend(y_pred)
            y_true_lst.extend(y_true)

    accuracy = metrics.accuracy_score(y_true_lst, y_pred_lst)
    false_pr, true_pr, thresholds = metrics.roc_curve(y_true_lst, probs_lst, pos_label = 1)
    auc = metrics.auc(false_pr, true_pr)
    balanced_acc = metrics.balanced_accuracy_score(y_true_lst, y_pred_lst)
    precision = metrics.precision_score(y_true_lst, y_pred_lst)
    recall = metrics.recall_score(y_true_lst, y_pred_lst)
    f1 = metrics.f1_score(y_true_lst, y_pred_lst)
    brier_score = metrics.brier_score_loss(y_true_lst, y_pred_lst)

    return [accuracy, balanced_acc, precision, recall, f1, brier_score, auc]


#Class use for logging console outputs
class Logger():
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

    def close(self):
        self.log.close()


#function for plotting losses
def plot_loss(training_mode, epoch_losses, res_path):
    print('Plotting losses...')
    num_epochs = len(epoch_losses[0])

    x = range(1, num_epochs + 1)

    plt.plot(x, epoch_losses[0], label = 'classifier loss')
    plt.plot(x, epoch_losses[1], label = 'transfer loss')
    plt.plot(x, epoch_losses[4], label = 'total loss')

    if training_mode == 1:
        plt.plot(x, epoch_losses[2], label = 'fisher loss')
        plt.plot(x, epoch_losses[3], label = 'EM loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Progression')
    plt.legend()
    plt.savefig(res_path + '/loss.pdf')
    print('Plot saved in ' + res_path +  '/loss.pdf')
    

def domain_cls_accuracy(d_out):
    '''Domain classification accuracy adapted from: https://github.com/HKUST-KnowComp/FisherDA/blob/master/src/utils.py.
    Args: 
        d_out: torch.FloatTensor, output of the domain classification network
    Returns:
        d0_acc: float, domain classification accuracy of both domains
        source_acc: float, domain classification accuracy of the source domain
        target_acc: float, domain classification accuracy of the target domain
    '''
    batch_size = d_out.size(0) // 2
    d0_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    d0_pred = d_out.detach() >= 0.5
    d0_acc = float(d0_pred.data.cpu().float().eq(d0_target).sum()) / float(d_out.size(0))
    source_acc = float(d0_pred[:batch_size].cpu().float().eq(d0_target[:batch_size]).sum()) / float(batch_size)
    target_acc = float(d0_pred[batch_size:].cpu().float().eq(d0_target[batch_size:]).sum()) / float(batch_size)
    
    return d0_acc, source_acc, target_acc
