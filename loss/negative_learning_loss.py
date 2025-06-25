import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class NegativeLearningLoss(nn.Module):
    def __init__(self, threshold=0.05):
        super(NegativeLearningLoss, self).__init__()
        self.threshold = threshold

    def forward(self, predict):
        mask = (predict < self.threshold).detach()
        negative_loss_item = -1 * mask * torch.log(1 - predict + 1e-6)
        negative_loss = torch.sum(negative_loss_item) / torch.sum(mask)

        # pdb.set_trace()

        return negative_loss



class Pseudo_label_Loss(nn.Module):
    def __init__(self, threshold=0.5):
        super(Pseudo_label_Loss, self).__init__()
        self.threshold = threshold

    def forward(self, predict):
        mask = (predict > self.threshold).detach()
        pseudo_loss_item = -1 * mask * torch.log(predict + 1e-6)
        pseudo_loss = torch.sum(pseudo_loss_item) / torch.sum(mask)

        # pdb.set_trace()

        return pseudo_loss
