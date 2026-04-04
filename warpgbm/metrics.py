# warpgbm/metrics.py

import torch

def rmsle_torch(y_true, y_pred, eps=1e-7):
    y_true = torch.clamp(y_true, min=0)
    y_pred = torch.clamp(y_pred, min=0)
    log_true = torch.log1p(y_true + eps)
    log_pred = torch.log1p(y_pred + eps)
    return torch.sqrt(torch.mean((log_true - log_pred) ** 2))

def softmax(logits, dim=-1):
    """Numerically stable softmax"""
    exp_logits = torch.exp(logits - torch.max(logits, dim=dim, keepdim=True)[0])
    return exp_logits / torch.sum(exp_logits, dim=dim, keepdim=True)

def log_loss_torch(y_true_labels, y_pred_probs, eps=1e-15):
    """
    Compute log loss (cross-entropy) for multiclass classification
    
    Args:
        y_true_labels: 1D tensor of true class labels (integers)
        y_pred_probs: 2D tensor of predicted probabilities [n_samples, n_classes]
        eps: Small value to clip probabilities for numerical stability
    """
    y_pred_probs = torch.clamp(y_pred_probs, eps, 1 - eps)
    n_samples = y_true_labels.shape[0]
    
    # Get the predicted probability for the true class
    true_class_probs = y_pred_probs[torch.arange(n_samples), y_true_labels.long()]
    
    # Return negative log likelihood
    return -torch.mean(torch.log(true_class_probs))

def accuracy_torch(y_true_labels, y_pred_labels):
    """Compute accuracy"""
    return (y_true_labels == y_pred_labels).float().mean()


def _normalize_weights(weights, reference):
    if weights is None:
        return torch.ones_like(reference, dtype=torch.float32)
    return weights.to(dtype=torch.float32)


def weighted_mse_torch(y_true, y_pred, sample_weight=None):
    weights = _normalize_weights(sample_weight, y_true)
    err2 = (y_true - y_pred) ** 2
    return torch.sum(weights * err2) / torch.sum(weights)


def weighted_multi_mse_torch(y_true, y_pred, sample_weight=None, valid_mask=None):
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(
            "weighted_multi_mse_torch expects 2D tensors [n_samples, n_outputs]."
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have matching shape, got {y_true.shape} and {y_pred.shape}."
        )

    if valid_mask is None:
        valid_mask = torch.isfinite(y_true)
    if valid_mask.shape != y_true.shape:
        raise ValueError(
            f"valid_mask must match y_true shape, got {valid_mask.shape} and {y_true.shape}."
        )

    weights = _normalize_weights(sample_weight, y_true).to(dtype=torch.float32)
    if weights.ndim == 1:
        weights = weights.view(-1, 1)
    if weights.shape != y_true.shape:
        raise ValueError(
            f"sample_weight must be shape {y_true.shape} or {(y_true.shape[0],)}, got {weights.shape}."
        )

    valid_mask_f = valid_mask.to(dtype=torch.float32)
    err2 = torch.where(valid_mask, (y_true - y_pred) ** 2, torch.zeros_like(y_pred))
    effective_weights = weights * valid_mask_f
    denom = torch.sum(effective_weights)
    if denom <= 0:
        raise ValueError("No valid weighted targets available for multi-output MSE.")
    return torch.sum(effective_weights * err2) / denom


def weighted_rmsle_torch(y_true, y_pred, sample_weight=None, eps=1e-7):
    weights = _normalize_weights(sample_weight, y_true)
    y_true = torch.clamp(y_true, min=0)
    y_pred = torch.clamp(y_pred, min=0)
    log_true = torch.log1p(y_true + eps)
    log_pred = torch.log1p(y_pred + eps)
    sq_err = (log_true - log_pred) ** 2
    return torch.sqrt(torch.sum(weights * sq_err) / torch.sum(weights))


def weighted_log_loss_torch(y_true_labels, y_pred_probs, sample_weight=None, eps=1e-15):
    y_pred_probs = torch.clamp(y_pred_probs, eps, 1 - eps)
    n_samples = y_true_labels.shape[0]
    true_class_probs = y_pred_probs[
        torch.arange(n_samples, device=y_true_labels.device), y_true_labels.long()
    ]
    weights = _normalize_weights(sample_weight, y_true_labels)
    losses = -torch.log(true_class_probs)
    return torch.sum(weights * losses) / torch.sum(weights)


def weighted_accuracy_torch(y_true_labels, y_pred_labels, sample_weight=None):
    weights = _normalize_weights(sample_weight, y_true_labels)
    correct = (y_true_labels == y_pred_labels).float()
    return torch.sum(weights * correct) / torch.sum(weights)


def weighted_corr_loss_torch(y_true, y_pred, sample_weight=None, eps=1e-12):
    weights = _normalize_weights(sample_weight, y_true)
    weight_sum = torch.sum(weights)
    mean_true = torch.sum(weights * y_true) / weight_sum
    mean_pred = torch.sum(weights * y_pred) / weight_sum

    centered_true = y_true - mean_true
    centered_pred = y_pred - mean_pred

    cov = torch.sum(weights * centered_true * centered_pred) / weight_sum
    var_true = torch.sum(weights * centered_true * centered_true) / weight_sum
    var_pred = torch.sum(weights * centered_pred * centered_pred) / weight_sum

    corr = cov / torch.sqrt(torch.clamp(var_true * var_pred, min=eps))
    return 1.0 - corr
