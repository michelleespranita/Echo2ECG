from typing import Optional

import torch

from util.metrics import RegressionMetrics, BinaryClassificationMetrics, MulticlassClassificationMetrics, MultilabelClassificationMetrics

def init_metrics_fn(task_type: str, device: Optional[torch.device] = None, num_outputs: int = None, num_tasks: int = None):
    if device is None:
        device = torch.device('cpu')

    if task_type == 'regression':
        return RegressionMetrics(device, num_outputs)
    elif task_type == 'binary_classification':
        return BinaryClassificationMetrics(device)
    elif task_type == 'multiclass_classification':
        return MulticlassClassificationMetrics(device, num_outputs)
    elif task_type == 'multilabel_classification':
        return MultilabelClassificationMetrics(device, num_outputs)
    else:
        raise NotImplementedError