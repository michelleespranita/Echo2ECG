from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.functional import confusion_matrix

from util.init_metrics import init_metrics_fn

class LinearProbe:
    def __init__(
        self,
        task_type: str = 'regression',
        device: torch.device = torch.device('cuda'),
        num_classes: int = None,
        save_dir: str = None
    ):
        self.task_type = task_type
        self.device = device
        self.num_classes = num_classes
        self.save_dir = save_dir

        if self.task_type == 'regression':
            self.model = LinearRegression()
        
        elif self.task_type in ['binary_classification', 'multiclass_classification']:
            self.model = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
        
        elif self.task_type == 'multilabel_classification':
            self.models = [KNeighborsClassifier(n_neighbors=5, algorithm='brute') for _ in range(num_classes)]
        
        self.metrics_fn = init_metrics_fn(self.task_type, self.device, self.num_classes)
        
    def _extract_features(self, dataloader, feature_fn):
        features, labels = None, None
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Linear probe - feature extraction - {self.task_type}'):
            out = feature_fn(batch)
            # out['global_token'] = F.normalize(out['global_token'], dim=-1)
            feat = out['global_token'].detach().cpu().numpy()
            label = out['label']

            if features is None:
                features = feat
                labels = label
            else:
                features = np.concatenate((features, feat), axis=0)
                labels = np.concatenate((labels, label), axis=0)
        
        return features, labels
    
    def fit(self, dataloader_train, feature_fn):
        features, labels = self._extract_features(dataloader_train, feature_fn)

        if self.task_type == 'multiclass_classification':
            labels = np.argmax(labels, axis=-1)
        
        if self.task_type in ['regression', 'binary_classification', 'multiclass_classification']:
            self.model.fit(features, labels)
        elif self.task_type == 'multilabel_classification':
            for class_idx in range(self.num_classes):
                self.models[class_idx].fit(features, labels[:, class_idx])
                
    def evaluate(self, dataloader_test, feature_fn, split_name):
        features, labels = self._extract_features(dataloader_test, feature_fn)

        if self.task_type == 'regression':
            preds = self.model.predict(features)

            preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
            labels = torch.tensor(labels, dtype=torch.float32, device=self.device)

            self.metrics_fn.update(preds, labels)
        
        elif self.task_type == 'binary_classification':
            probs = self.model.predict_proba(features) # (B, 2)
            
            probs = torch.tensor(probs, dtype=torch.float32, device=self.device)
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)

            probs = probs[:, 1] # select the positive class

            for name, metric in self.metrics_fn.metrics.items():
                if name in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']:
                    preds = (probs > 0.5).long()
                elif name in ['auroc', 'auprc']:
                    preds = probs
                metric.update(preds, labels)
                
        elif self.task_type == 'multiclass_classification':
            probs = self.model.predict_proba(features) # (B, num_classes)

            probs = torch.tensor(probs, dtype=torch.float32, device=self.device)
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)

            preds = probs
            labels = labels.argmax(dim=-1)

            for name, metric in self.metrics_fn.metrics.items():
                metric.update(preds, labels)
        
        elif self.task_type == 'multilabel_classification':
            
            all_probs = []
            for class_idx in range(self.num_classes):
                probs = self.models[class_idx].predict_proba(features) # (B, 2)
            
                probs = torch.tensor(probs, dtype=torch.float32, device=self.device)

                probs = probs[:, 1] # select the positive class

                all_probs.append(probs)
            
            probs = torch.stack(all_probs, dim=-1)
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)

            for name, metric in self.metrics_fn.metrics.items():
                if name in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']:
                    preds = (probs > 0.5).long()
                elif name in ['auroc', 'auprc']:
                    preds = probs
                metric.update(preds, labels)
                        
        metrics = self.metrics_fn.compute()

        # save predictions
        if self.save_dir is not None:
            self._save_predictions(preds, labels, split_name)
            
        return metrics
    
    def _save_predictions(self, preds, labels, split_name):
        if self.task_type == 'binary_classification':
            cm = confusion_matrix(preds, labels, task='binary')
        elif self.task_type == 'multilabel_classification':
            cm = confusion_matrix(preds, labels, task='multilabel', num_labels=self.num_classes)
        elif self.task_type == 'multiclass_classification':
            cm = confusion_matrix(preds, labels, task='multiclass', num_classes=self.num_classes)
        else:
            cm = None
        
        print('Confusion matrix', cm)
        
        if cm is not None:
            torch.save(cm, os.path.join(self.save_dir, f'confmat_{split_name}.pt'))

        # Save raw predictions and argmaxed predictions
        if self.task_type != 'regression':
            torch.save(preds, os.path.join(self.save_dir, f'preds_{split_name}.pt'))