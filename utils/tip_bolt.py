import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            epoch = self.last_epoch - self.warmup_epochs
            total_epochs = self.max_epochs - self.warmup_epochs
            return [base_lr * (1 + math.cos(math.pi * epoch / total_epochs)) / 2 for base_lr in self.base_lrs]