import torch
import torch.nn.functional as F

def masked_mse_loss(predictions, targets, mask):
    """Masked MSE Loss to ignore NaN affected stocks"""
    loss = (predictions - targets) ** 2  # Element-wise MSE
    loss = loss * mask  # Apply mask
    return loss.sum() / mask.sum()  # Normalize by valid elements

def weighted_mask_mse_loss(predictions, targets, mask):
    """Weighted Masked MSE Loss: larger targets have higher weight.
    
    Each squared error is weighted by the absolute value of the target.
    The loss is computed only for valid (masked) elements and normalized 
    by the sum of weights of those elements.
    """
    # Calculate element-wise squared error
    loss = (predictions - targets) ** 2
    
    # Define weights based on target magnitude (using absolute value)
    weights = abs(targets)
    
    # Apply the mask and weight
    weighted_loss = loss * weights * mask
    
    # Normalize by the sum of weights for the valid elements.
    normalization = (weights * mask).sum()
    
    return weighted_loss.sum() / normalization

def thresh_mask_mse_loss(predictions, targets, mask, threshs):
    """
    Masked MSE Loss with threshold-based penalty.

    If predictions are above thresh while targets are below, or vice versa,
    the loss is doubled. Otherwise, normal MSE is applied.
    
    Parameters:
    - predictions: Tensor of predicted values.
    - targets: Tensor of actual values.
    - mask: Binary tensor (0 for NaN/missing, 1 for valid data).
    - threshs: Tensor of threshold values for each data point.

    Returns:
    - Scaled masked MSE loss.
    """
    loss = (predictions - targets) ** 2  # Element-wise MSE

    # Identify cases where prediction and target are on opposite sides of the threshold
    mismatch_condition = ((predictions > threshs) & (targets < threshs)) | \
                         ((predictions < threshs) & (targets > threshs))

    # Double the loss where mismatch condition is met
    loss = torch.where(mismatch_condition, loss * 2, loss)

    # Apply mask to ignore invalid data
    loss = loss * mask  

    # Normalize by the count of valid elements
    return loss.sum() / mask.sum()


def masked_ccc_loss(predictions, targets, mask, eps=1e-8):
    """
    Masked Concordance Correlation Coefficient (CCC) Loss computed across dimension N.
    This version aggregates the loss to a scalar.

    Args:
        predictions (Tensor): Predicted values of shape (B, N).
        targets (Tensor): Target values of shape (B, N).
        mask (Tensor): Binary mask of shape (B, N), where 1 indicates a valid element.
        eps (float): A small constant to prevent division by zero.

    Returns:
        Tensor: A scalar loss value.
    """
    # Compute the sum of valid (masked) elements for each sample.
    sum_mask = mask.sum(dim=1)  # shape: (B,)

    # Compute the masked means for predictions and targets.
    mean_pred = (predictions * mask).sum(dim=1) / (sum_mask + eps)  # shape: (B,)
    mean_target = (targets * mask).sum(dim=1) / (sum_mask + eps)      # shape: (B,)

    # Compute the masked variances for predictions and targets.
    var_pred = (mask * (predictions - mean_pred.unsqueeze(1)) ** 2).sum(dim=1) / (sum_mask + eps)
    var_target = (mask * (targets - mean_target.unsqueeze(1)) ** 2).sum(dim=1) / (sum_mask + eps)

    # Compute the masked covariance between predictions and targets.
    cov = (mask * (predictions - mean_pred.unsqueeze(1)) * (targets - mean_target.unsqueeze(1))).sum(dim=1) / (sum_mask + eps)

    # Compute the Concordance Correlation Coefficient (CCC) for each sample.
    ccc = (2 * cov) / (var_pred + var_target + (mean_pred - mean_target) ** 2 + eps)

    # Define the loss as 1 - CCC for each sample and average over the batch.
    loss = 1 - ccc  # shape: (B,)
    return loss.mean()


def min_max_weighted_mask_mse_loss(predictions, targets, mask, eps=1e-6):
    # squared error
    se = (predictions - targets) ** 2


    # get min/max over valid elements
    t = targets * mask
    t_min = t[mask.bool()].min()
    t_max = t[mask.bool()].max()


    # min–max scale into [0,1]
    weights = (targets - t_min) / (t_max - t_min + eps)


    # optional: shift into [1,2] so no zero-weight
    weights = 1.0 + weights


    # apply mask
    weighted_se = se * weights * mask
    norm = (weights * mask).sum()


    return weighted_se.sum() / (norm + eps)


def min_max_reverse_weighted_mask_mse_loss(predictions, targets, mask, eps=1e-6):
    # this is for min_price prediction
    # squared error
    se = (predictions - targets) ** 2


    # get min/max over valid elements
    t = targets * mask
    t_min = t[mask.bool()].min()
    t_max = t[mask.bool()].max()


    # min–max scale into [0,1]
    weights = (targets - t_min) / (t_max - t_min + eps)
    # reverse the weights
    weights = 1.0 - weights

    # optional: shift into [1,2] so no zero-weight
    weights = 1.0 + weights


    # apply mask
    weighted_se = se * weights * mask
    norm = (weights * mask).sum()


    return weighted_se.sum() / (norm + eps)


def exp_weighted_mask_mse_exp(predictions, targets, mask, beta=1.0, eps=1e-6):
    se = (predictions - targets) ** 2


    # exponential weighting
    weights = torch.exp(beta * targets)


    weighted_se = se * weights * mask
    norm = (weights * mask).sum()


    return weighted_se.sum() / (norm + eps)

def masked_rank_loss(predictions: torch.Tensor,
                     targets:     torch.Tensor,
                     mask:        torch.Tensor,
                     margin:      float = 1.0,
                     eps:         float = 1e-8) -> torch.Tensor:
    """
    Masked Pairwise Hinge Ranking Loss across N for each B, then average over batch.

    Args:
        predictions (Tensor): shape (B, N)
        targets     (Tensor): shape (B, N)
        mask        (Tensor): binary mask (0/1) shape (B, N)
        margin      (float): margin for hinge loss
        eps         (float): small constant to avoid 0/0

    Returns:
        Tensor: scalar rank loss
    """
    B, N = predictions.shape
    losses = []
    for b in range(B):
        # 選出 valid indices
        valid = mask[b].bool()
        pred_b   = predictions[b, valid]  # shape (M,)
        targ_b   = targets[b,    valid]  # shape (M,)
        M = pred_b.size(0)
        # 少於兩個元素就跳過
        if M < 2:
            continue
        
        t_min, t_max = targ_b.min(), targ_b.max()
        if (t_max - t_min) > eps:
            targ_b = (targ_b - t_min) / (t_max - t_min)
            pred_b = (pred_b - t_min) / (t_max - t_min)
        else:
            # 若全部 target 相同，跳過
            continue

        # 計算 pairwise target 差值
        # 若 targ_b[i] > targ_b[j]，則 pos_pairs[i,j]=True
        targ_diff = targ_b.unsqueeze(1) - targ_b.unsqueeze(0)  # (M, M)
        pos_pairs = targ_diff > eps                              # (M, M) bool

        # 如果沒有正例對，也跳過
        if not pos_pairs.any():
            continue

        # 計算 pairwise prediction 差值
        pred_diff = pred_b.unsqueeze(1) - pred_b.unsqueeze(0)    # (M, M)
        
        weights = torch.relu(targ_diff)
        # hinge loss: max(0, margin - pred_diff) 針對正例對
        pairwise_loss = F.relu(margin - pred_diff) * weights               # (M, M)
        # 只選正例對的位置
        loss_b = pairwise_loss[pos_pairs].sum() / (weights[pos_pairs].sum() + eps)
        losses.append(loss_b)

    if not losses:
        # 若整批都沒有效 pair，回傳 0
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    # 最後對每個樣本的 loss 取平均
    return torch.stack(losses).mean()

# TODO: Berlin (寫 cross entropy loss)
def masked_cross_entropy_loss(predictions, targets, mask):
    raise NotImplementedError("Cross entropy loss is not implemented yet.")