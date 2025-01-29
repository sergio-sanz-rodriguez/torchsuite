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
        y_score (Tensor): Predicted scores/probabilities for each class, shape [batch_size, num_classes].

    Returns:
        fpr (Tensor): False positive rate at each threshold.
        tpr (Tensor): True positive rate at each threshold.
    """
    # Sort the predictions in descending order and get the indices
    sorted_scores, sorted_indices = torch.sort(y_score, descending=True, dim=0)
    
    # Initialize variables
    tpr = torch.zeros(sorted_scores.size(0), device=y_score.device)
    fpr = torch.zeros(sorted_scores.size(0), device=y_score.device)
    total_positive = torch.sum(y_true).item()
    #total_negative = y_true.size(0) - total_positive
    
    # Loop through each threshold and compute TPR and FPR
    for i in range(sorted_scores.size(0)):
        threshold = sorted_scores[i]
        
        # Get predictions for this threshold
        predictions = (y_score >= threshold).float()
        
        # Compute True Positives, False Positives, True Negatives, False Negatives
        tp = torch.sum((predictions == 1) & (y_true == 1)).item()
        fp = torch.sum((predictions == 1) & (y_true == 0)).item()
        tn = torch.sum((predictions == 0) & (y_true == 0)).item()
        fn = torch.sum((predictions == 0) & (y_true == 1)).item()
        
        # Compute TPR and FPR
        tpr[i] = tp / (tp + fn) if tp + fn > 0 else 0
        fpr[i] = fp / (fp + tn) if fp + tn > 0 else 0
    
    return fpr, tpr

class SmoothPAUCLossMulticlass(torch.nn.Module):
    def __init__(self, recall_range=(0.95, 1.0), lambda_fpr=1.0, num_classes=101, label_smoothing=0.1):
        super(SmoothPAUCLossMulticlass, self).__init__()
        self.recall_range = recall_range
        self.max_pauc = recall_range[1] - recall_range[0]
        self.lambda_fpr = lambda_fpr
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, predictions, targets):
        """
        Custom loss to optimize Partial Area Under the ROC Curve (pAUC) in the specified recall range for multi-class classification.
        Incorporates label smoothing in the Cross-Entropy loss.
        """
        # Convert to probabilities for each class using softmax
        probs = F.softmax(predictions, dim=1)  # Shape: [batch_size, num_classes]
        
        # Convert targets to one-hot encoding
        targets = targets.to(predictions.device)
        targets_one_hot = torch.eye(self.num_classes, device=predictions.device)[targets]

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = (1 - self.label_smoothing) * targets_one_hot + self.label_smoothing / self.num_classes
        
        # Calculate AUC for each class
        p_auc_values = []
        for i in range(self.num_classes):
            # Get the probabilities and ground truth for class i
            class_probs = probs[:, i]
            class_targets = targets_one_hot[:, i]
            
            # Compute ROC curve on GPU
            fpr_vals, tpr_vals = roc_curve_gpu(class_targets, class_probs)

            # Compute the mask for the desired recall range (e.g., recall between 0.8 and 1.0)
            recall_mask = (tpr_vals >= self.recall_range[0]) & (tpr_vals <= self.recall_range[1])
            
            # Check if the recall_mask has any valid entries
            if recall_mask.sum() > 0:
                # Compute partial AUC for class i
                p_auc_class = auc(fpr_vals[recall_mask].cpu(), tpr_vals[recall_mask].cpu())  # AUC expects numpy, so move to CPU
                p_auc_values.append(p_auc_class)
            else:
                print(f"No valid recall points found for class {i}. Skipping AUC calculation.")
                p_auc_values.append(0.0)  # Assign a default value if no valid points are found

        # Average pAUC across all classes
        avg_p_auc = np.mean(p_auc_values) / self.max_pauc

        # Compute the Cross-Entropy Loss with label smoothing
        log_probs = F.log_softmax(predictions, dim=1)  # Log probabilities
        ce_loss = -(targets_one_hot * log_probs).sum(dim=1).mean()

        # Total Loss: Subtract the average pAUC from the CE loss to encourage higher pAUC
        total_loss = ce_loss - self.lambda_fpr * (avg_p_auc - 1)

        return total_loss