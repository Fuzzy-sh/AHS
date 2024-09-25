import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(CosineSimilarityLoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        cosine_loss = 1 - F.cosine_similarity(output, target, dim=-1).mean()
        return self.alpha * cosine_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy_loss = nn.NLLLoss()
        self.cosine_similarity_loss = CosineSimilarityLoss()

    def forward(self, output, target, labels):
        ce_loss = self.cross_entropy_loss(output, labels)
        cs_loss = self.cosine_similarity_loss(output, target)
        return self.alpha * ce_loss + self.beta * cs_loss

