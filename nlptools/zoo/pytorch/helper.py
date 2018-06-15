"""
Use in PyTorch.
"""

def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum() / target.numel()
    return acc


class BinaryClassificationMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.pre = 0
        self.rec = 0
        self.f1 = 0

    def update(self, output, target):
        pred = output >= 0.5
        truth = target >= 0.5
        self.tp += pred.mul(truth).sum(0).float()
        self.tn += (1 - pred).mul(1 - truth).sum(0).float()
        self.fp += pred.mul(1 - truth).sum(0).float()
        self.fn += (1 - pred).mul(truth).sum(0).float()
        self.acc = (self.tp + self.tn).sum() / (self.tp + self.tn + self.fp + self.fn).sum()
        self.pre = self.tp / (self.tp + self.fp)
        self.rec = self.tp / (self.tp + self.fn)
        self.f1 = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)
        self.avg_pre = nanmean(self.pre)
        self.avg_rec = nanmean(self.rec)
        self.avg_f1 = nanmean(self.f1)