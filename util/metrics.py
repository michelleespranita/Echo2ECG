from typing import Optional

import torch
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC, AveragePrecision
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, MultilabelPrecision,\
    MultilabelRecall, MultilabelAUROC, MultilabelAveragePrecision

class Metrics():
    def __init__(
        self,
        device: Optional[torch.device] = torch.device('cpu')
    ):
        self.device = device
        self.metrics = {}
    
    def update(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        for name, metric in self.metrics.items():
            pred_proc, label_proc = self._prepare_inputs(name, pred, label)
            metric.update(pred_proc, label_proc)
        
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor):
        # Default: do nothing
        return pred, label
    
    def compute(self) -> dict:
        metrics_dict = {}
        for metric_name, metric in self.metrics.items():
            metrics_dict[metric_name] = metric.compute()
        return metrics_dict
    
    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
    
    def get(self, name: str) -> float:
        if name not in self.metrics:
            raise ValueError(f"Metric {name} not found. Available: {list(self.metrics.keys())}")
        return self.metrics[name].compute()

class RegressionMetrics(Metrics):
    def __init__(
        self,
        device: Optional[torch.device] = torch.device('cpu'),
        num_outputs: int = None
    ):
        super().__init__(device)

        if num_outputs is None:
            self.num_outputs = 1
        else:
            self.num_outputs = num_outputs

        self.metrics = {
            'mae': MeanAbsoluteError().to(device),
            'mse': MeanSquaredError().to(device),
            'r2': R2Score(self.num_outputs, multioutput='uniform_average').to(device),
            'r2_per_output': R2Score(self.num_outputs, multioutput='raw_values').to(device),
        }

class BinaryClassificationMetrics(Metrics):
    def __init__(
        self,
        device = torch.device('cpu')
    ):
        super().__init__(device)

        self.metrics = {
            'accuracy': Accuracy(task='binary').to(device),
            'balanced_accuracy': Accuracy(task='multiclass', average='macro', num_classes=2).to(device),
            'f1': F1Score(task='binary').to(device),
            'precision': Precision(task='binary').to(device),
            'recall': Recall(task='binary').to(device),
            'auroc': AUROC(task='binary').to(device),
            'auprc': AveragePrecision(task='binary').to(device),
        }
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor): # pred: logits
        probs = torch.sigmoid(pred)
        label = label.long()

        if name in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']:
            pred = (probs > 0.5).long()
        elif name in ['auroc', 'auprc']:
            pred = probs
        
        return pred, label

class MulticlassClassificationMetrics(Metrics):
    def __init__(
        self,
        device = torch.device('cpu'),
        num_classes: int = None
    ):
        super().__init__(device)

        if num_classes is None:
            self.num_classes = 1
        else:
            self.num_classes = num_classes

        self.metrics = {
            'accuracy': Accuracy(task='multiclass', average='macro', num_classes=self.num_classes).to(device), # balanced accuracy
            'f1': F1Score(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'precision': Precision(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'recall': Recall(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'auroc': AUROC(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'auprc': AveragePrecision(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
        }
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor): # pred: logits
        # logits -> probabilities
        probs = torch.softmax(pred, dim=-1)
        pred = probs
        label = label.long()
        label = label.argmax(dim=-1)

        return pred, label

class MultilabelClassificationMetrics(Metrics):
    def __init__(
        self,
        device = torch.device('cpu'),
        num_labels: int = None
    ):
        super().__init__(device)
        
        if num_labels is None:
            self.num_labels = 1
        else:
            self.num_labels = num_labels

        self.metrics = {
            'accuracy': MultilabelAccuracy(num_labels=self.num_labels, average='macro').to(device),
            'f1': MultilabelF1Score(num_labels=self.num_labels, average='macro').to(device),
            'precision': MultilabelPrecision(num_labels=self.num_labels, average='macro').to(device),
            'recall': MultilabelRecall(num_labels=self.num_labels, average='macro').to(device),
            'auroc': MultilabelAUROC(num_labels=self.num_labels, average='macro').to(device),
            'auroc_per_output': MultilabelAUROC(num_labels=self.num_labels, average='none').to(device),
            'auprc': MultilabelAveragePrecision(num_labels=self.num_labels, average='macro').to(device),
        }
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor):
        # logits -> probs via sigmoid
        probs = torch.sigmoid(pred)
        label = label.long()

        if name in ['accuracy', 'f1', 'precision', 'recall']:
            pred = (probs > 0.5).long()  # thresholded preds
        elif name in ['auroc', 'auprc']:
            pred = probs

        return pred, label