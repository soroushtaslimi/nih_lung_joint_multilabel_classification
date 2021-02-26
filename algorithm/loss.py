import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduction='none')

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)




def loss_jocor(y_1, y_2, t, forget_rate, ind, noise_or_not=None, co_lambda=0.1, class_weights=None):

    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).cuda()

    # loss_pick_1 = F.cross_entropy(y_1, t, reduction='none', weight=class_weights) * (1-co_lambda)
    # loss_pick_2 = F.cross_entropy(y_2, t, reduction='none', weight=class_weights) * (1-co_lambda)
    t = t.type_as(y_1)  # for bceloss type of t should be float
    loss_pick_1 = F.binary_cross_entropy_with_logits(y_1, t, reduction='none', weight=class_weights) * (1-co_lambda)
    loss_pick_2 = F.binary_cross_entropy_with_logits(y_2, t, reduction='none', weight=class_weights) * (1-co_lambda)
    loss_pick_1 = torch.mean(loss_pick_1, dim=1)
    loss_pick_2 = torch.mean(loss_pick_2, dim=1)
    """
    print('t.shape, y_1.shape, y_2.shape:', t.shape, y_1.shape, y_2.shape)
    print('loss_pick_1.shape:', loss_pick_1.shape)
    print('loss_pick_2.shape:', loss_pick_2.shape)
    print('kl_loss_compute(y_1, y_2,reduce=False).shape:', kl_loss_compute(y_1, y_2,reduce=False).shape)
    print('kl_loss_compute(y_2, y_1,reduce=False).shape:', kl_loss_compute(y_2, y_1,reduce=False).shape)
    """
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False)).cpu()


    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    if noise_or_not is None:
        pure_ratio = 0.0
    else:
        pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

    ind_update=ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return loss, loss, pure_ratio, pure_ratio


