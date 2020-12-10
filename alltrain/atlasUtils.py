import medpy.metric.binary as medpyMetrics
import numpy as np
import math
import torch

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

def dice(pred, target):
    predBin = (pred > 0.5).float()
    # predBin = (pred > 0.5)
    # predBin = predBin.float()
    return softDice(predBin, target, 0, True).item()

def diceLoss(pred, target, nonSquared=False):
    return 1 - softDice(pred, target, nonSquared=nonSquared)

def atlasDiceLoss(outputs, labels, nonSquared=False):
    # print('###IN LOSS###')
    # print('outputs', outputs.sum().cpu().item())
    # print('labels.sum', labels.sum().cpu().item())
    # print('labels.max', labels.max().cpu().item())
    # print('labels.min', labels.min().cpu().item())
    #bring outputs into correct shape
    n_classe = 13
    chunk = list(outputs.chunk(n_classe, dim=1))
    
    s = chunk[0].shape
    

    for i in range(n_classe):
        # print('chunk['+str(i)+'].sum', chunk[i].sum().cpu().item())
        chunk[i] = chunk[i].view(s[0], s[2], s[3], s[4])
        # print('view.chunk['+str(i)+'].sum', chunk[i].sum().cpu().item())
        # print('view.chunk['+str(i)+'].shape', chunk[i].shape)
        


    # bring masks into correct shape
    chunkMask = list(labels.chunk(n_classe, dim=1))
    # print('len(chunkMask)', len(chunkMask))
    # print('len(chunk)', len(chunk))
    s = chunkMask[0].shape
    
    for i in range(n_classe):
        # print('chunkMask['+str(i)+'].sum', chunkMask[i].sum().cpu().item())
        chunkMask[i] = chunkMask[i].view(s[0], s[2], s[3], s[4])
        # print('view.chunkMask['+str(i)+'].sum', chunkMask[i].sum().cpu().item())
        # print('view.chunkMask['+str(i)+'].shape', chunkMask[i].shape)
        


    

    #calculate losses
    losses = []
    for i in range(n_classe):
        losses.append(diceLoss(chunk[i], chunkMask[i], nonSquared=nonSquared))

    # print('###END LOSS###')

    return sum(losses) / n_classe

def MyAtlasDiceLoss(outputs, labels, nonSquared=False):
    n_classe = 13
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
    return (labels == i).half()

# def getTCMask(labels):
#     return ((labels != 0) * (labels != 2)).float() #We use multiplication as AND

# def getETMask(labels):
#     return (labels == 4).float()
