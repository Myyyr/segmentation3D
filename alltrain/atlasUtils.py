import medpy.metric.binary as medpyMetrics
import numpy as np
import math
import torch
import torch.nn.functional as F

def softDice(pred, target, smoothing=1, nonSquared=False):

    intersection = (pred * target).sum(dim=(1, 2, 3))
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def softDice2D(pred, target, smoothing=1, nonSquared=False):

    intersection = (pred * target).sum(dim=(1, 2))
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum(dim=(1, 2)) + (target * target).sum(dim=(1, 2))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def dice(pred, target):
    # predBin = (pred > 0.5).float()
    if len(pred.shape) == 4:
        return softDice(pred.float(), target, 1e-7, True).item()
    elif len(pred.shape) == 3:
        return softDice2D(pred.float(), target, 1e-7, True).item()

def diceLoss(pred, target, nonSquared=False):
    return 1 - softDice(pred, target, smoothing=0.0001, nonSquared=nonSquared)




def atlasDiceLoss(outputs, labels, nonSquared=False, n_classe = 14):
    #bring outputs into correct shape
    outputs = F.softmax(outputs, dim=1)
    # print(np.unique(outputs.argmax(dim = 1).detach().cpu().numpy()))
    chunk = list(outputs.chunk(n_classe, dim=1))
    s = chunk[0].shape
    for i in range(n_classe):
        chunk[i] = chunk[i].view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    chunkMask = list(labels.chunk(n_classe, dim=1))
    s = chunkMask[0].shape
    
    for i in range(n_classe):
        chunkMask[i] = chunkMask[i].view(s[0], s[2], s[3], s[4])


    

    #calculate losses
    losses = []
    for i in range(n_classe):
        losses.append(diceLoss(chunk[i], chunkMask[i], nonSquared=nonSquared))

    # print('###END LOSS###')

    return sum(losses) / n_classe

def MyAtlasDiceLoss(outputs, labels, nonSquared=False):
    n_classe = 14
    smooth = 0.01

    inter = torch.sum(outputs * labels) + smooth
    union = torch.sum(outputs) + torch.sum(outputs) + smooth

    dice = 2.0 * inter / union
    dice = 1.0 - dice/float(n_classe)

    return dice


def sensitivity(pred, target):
    predBin = (pred > 0.5).float()
    intersection = (predBin * target).sum()
    allPositive = target.sum()

    # special case for zero positives
    if allPositive == 0:
        return 1.0
    return (intersection / allPositive).item()

def specificity(pred, target):
    predBinInv = (pred <= 0.5).float()
    targetInv = (target == 0).float()
    intersection = (predBinInv * targetInv).sum()
    allNegative = targetInv.sum()
    return (intersection / allNegative).item()

def getHd95(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
        return hd95
    else:
        # Edge cases that medpy cannot handle
        return -1

def getMask(labels, i):
    # return (labels == i).float()
    return (labels == i)*1

# def getTCMask(labels):
#     return ((labels != 0) * (labels != 2)).float() #We use multiplication as AND

# def getETMask(labels):
#     return (labels == 4).float()
