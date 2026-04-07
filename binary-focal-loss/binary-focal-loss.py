import math

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    focal_loss = []
    for i in range(len(predictions)):
        p = predictions[i]
        p_t = p if targets[i] == 1 else 1-p
        fl = -alpha * (1-p_t)**gamma * math.log(p_t)
        focal_loss.append(fl)

    return sum(focal_loss)/len(focal_loss)
        