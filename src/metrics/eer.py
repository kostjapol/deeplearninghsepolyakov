import torch

from src.metrics.base_metric import BaseMetric


def compute_eer(scores, labels):
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()

    order = torch.argsort(scores)
    labels = labels[order]

    bonafide = (labels == 1).float()
    spoof = (labels == 0).float()

    false_reject = torch.cumsum(bonafide, dim=0) / bonafide.sum()
    false_accept = 1 - torch.cumsum(spoof, dim=0) / spoof.sum()

    index = torch.argmin(torch.abs(false_reject - false_accept))
    eer = (false_reject[index] + false_accept[index]) / 2

    return eer.item() * 100


class EERMetric(BaseMetric):
    def __init__(self, name="EER"):
        super().__init__(name=name)
        self.reset()

    def reset(self):
        self.scores = []
        self.labels = []

    def __call__(self, logits, labels, **batch):
        probabilities = torch.softmax(logits, dim=1)
        bonafide_scores = probabilities[:, 1]

        self.scores.append(bonafide_scores.detach().cpu())
        self.labels.append(labels.detach().cpu())

    def compute(self):
        scores = torch.cat(self.scores)
        labels = torch.cat(self.labels)

        return compute_eer(scores, labels)
