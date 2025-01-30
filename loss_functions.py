import torch
import torch.nn.functional as F
from sklearn.metrics import auc
import numpy as np

def roc_curve_gpu(y_true, y_score):
    """
    Compute ROC curve on GPU. This function calculates the True Positive Rate (TPR)
    and False Positive Rate (FPR) for various thresholds.

    Arguments:
        y_true (Tensor): Ground truth labels, shape [batch_size].
        y_score (Tensor): Predicted scores/probabilities for each class, shape [batch_size] (binary) or [batch_size, num_classes] (multiclass).

    Returns:
        fpr (Tensor): False positive rate at each threshold.
        tpr (Tensor): True positive rate at each threshold.
    """
    # Handle binary or multiclass cases
    if y_score.dim() == 1:
        # Binary case: Treat y_score as probabilities for the positive class
        sorted_scores, sorted_indices = torch.sort(y_score, descending=True)
        num_classes = 2
    else:
        # Multiclass case
        num_classes = y_score.size(1)
        fpr_list, tpr_list = [], []
        for i in range(num_classes):
            class_probs = y_score[:, i]
            class_true = (y_true == i).float()
            sorted_scores, sorted_indices = torch.sort(class_probs, descending=True)

            # Compute TPR and FPR for this class
            tpr = torch.zeros(sorted_scores.size(0), device=y_score.device)
            fpr = torch.zeros(sorted_scores.size(0), device=y_score.device)
            total_positive = torch.sum(class_true).item()
            total_negative = class_true.size(0) - total_positive

            for j in range(sorted_scores.size(0)):
                threshold = sorted_scores[j]
                predictions = (class_probs >= threshold).float()
                tp = torch.sum((predictions == 1) & (class_true == 1)).item()
                fp = torch.sum((predictions == 1) & (class_true == 0)).item()
                tn = torch.sum((predictions == 0) & (class_true == 0)).item()
                fn = torch.sum((predictions == 0) & (class_true == 1)).item()

                tpr[j] = tp / (tp + fn) if tp + fn > 0 else 0
                fpr[j] = fp / (fp + tn) if fp + tn > 0 else 0

            fpr_list.append(fpr)
            tpr_list.append(tpr)

        return torch.stack(fpr_list), torch.stack(tpr_list)

    # For binary case
    tpr = torch.zeros(sorted_scores.size(0), device=y_score.device)
    fpr = torch.zeros(sorted_scores.size(0), device=y_score.device)
    total_positive = torch.sum(y_true).item()
    total_negative = y_true.size(0) - total_positive

    for i in range(sorted_scores.size(0)):
        threshold = sorted_scores[i]
        predictions = (y_score >= threshold).float()
        tp = torch.sum((predictions == 1) & (y_true == 1)).item()
        fp = torch.sum((predictions == 1) & (y_true == 0)).item()
        tn = torch.sum((predictions == 0) & (y_true == 0)).item()
        fn = torch.sum((predictions == 0) & (y_true == 1)).item()

        tpr[i] = tp / (tp + fn) if tp + fn > 0 else 0
        fpr[i] = fp / (fp + tn) if fp + tn > 0 else 0

    return fpr, tpr

class CrossEntropyPAUCLoss(torch.nn.Module):

    """
    Custom loss combining Cross-Entropy and Partial AUC (pAUC) optimization for multi-class classification.
    
    Arguments:
        recall_range (tuple): Range for recall (True Positive Rate) used in pAUC calculation (start, end).
        lambda_pauc (float): Weight for the pAUC calculation or contribution of pAUC in the loss function: >= 0.0
        num_classes (int): Number of classes for classification.
        label_smoothing (float): Factor for label smoothing to reduce overfitting.
        weight (Tensor): Optional class weights for handling class imbalance.
    """

    def __init__(
            self,
            recall_range=(0.95, 1.0),
            lambda_pauc=1.0,
            num_classes=101,
            label_smoothing=0.1,
            weight=None):
        
        super(CrossEntropyPAUCLoss, self).__init__()
        self.recall_range = recall_range
        self.max_pauc = recall_range[1] - recall_range[0]
        self.lambda_pauc = lambda_pauc
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        # Register weight tensor (to handle class imbalance)
        if weight is not None:
            self.weight = weight.clone().detach().to(dtype=torch.float32)
        else:
            self.weight = torch.ones(num_classes, dtype=torch.float32)

    def forward(self, predictions, targets):

        """
        Computes the loss that combines Cross-Entropy and Partial AUC (pAUC) for multi-class classification.

        Arguments:
            predictions (Tensor): The model's raw output (logits), shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].

        Returns:
            Tensor: The computed total loss (Cross-Entropy + pAUC).
        """
        
        # Convert to probabilities for each class using softmax
        probs = F.softmax(predictions, dim=1)  # Shape: [batch_size, num_classes]

        # Convert targets to one-hot encoding
        targets = targets.to(predictions.device)
        targets_one_hot = torch.eye(self.num_classes, device=predictions.device)[targets]

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = (1 - self.label_smoothing) * targets_one_hot + self.label_smoothing / self.num_classes


        # Compute AUC for each class
        p_auc_values = []
        if self.num_classes == 2:
            class_probs = probs[:, 1]
            class_targets = (targets_one_hot[:, 1] > 0.5).float()  # Threshold at 0.5
            
            # Compute ROC curve (assumed function available)
            fpr_vals, tpr_vals = roc_curve_gpu(class_targets, class_probs)

            # Compute the mask for the recall range
            recall_mask = (tpr_vals >= self.recall_range[0]) & (tpr_vals <= self.recall_range[1])

            if recall_mask.sum() > 0:
                # Compute weighted partial AUC using trapezoidal rule
                pauc = torch.trapz(tpr_vals[recall_mask], fpr_vals[recall_mask])  # Approximate AUC
                p_auc_values.append(pauc * self.weight[1].to(predictions.device))
            else:
                p_auc_values.append(torch.tensor(0.0, device=predictions.device))
                
        else:
            for i in range(self.num_classes):
                class_probs = probs[:, i]
                class_targets = (targets_one_hot[:, i] > 0.5).float()  # Threshold at 0.5
                
                # Compute ROC curve (assumed function available)
                fpr_vals, tpr_vals = roc_curve_gpu(class_targets, class_probs)

                # Compute the mask for the recall range
                recall_mask = (tpr_vals >= self.recall_range[0]) & (tpr_vals <= self.recall_range[1])

                if recall_mask.sum() > 0:
                    # Compute weighted partial AUC using trapezoidal rule
                    pauc = torch.trapz(tpr_vals[recall_mask], fpr_vals[recall_mask])  # Approximate AUC
                    p_auc_values.append(pauc * self.weight[i].to(predictions.device))
                else:
                    p_auc_values.append(torch.tensor(0.0, device=predictions.device))

        # Weighted mean of pAUC across all classes
        avg_p_auc = torch.sum(torch.stack(p_auc_values)) / (self.weight.sum().to(predictions.device) * self.max_pauc)

        # Compute the weighted Cross-Entropy Loss with label smoothing
        log_probs = F.log_softmax(predictions, dim=1)
        ce_loss = -(targets_one_hot * log_probs * self.weight.to(predictions.device)).sum(dim=1).mean()

        # Total Loss: Subtract weighted pAUC from CE loss
        pauc_loss = -torch.log(avg_p_auc + 1e-7)
        total_loss = ce_loss + self.lambda_pauc * pauc_loss

        return total_loss