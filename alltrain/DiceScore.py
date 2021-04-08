import torch
import numpy as np

class DiceScore():
    def __init__(self, classes, epsilon = 1e-7):
        """
        classes : list : list of classe's names
        """
        self.n_classes = len(classes)
        self.classes = classes

        self.n_count = 0
        self.accum = {}
        self.all_dices = {}
        for k in self.classes : 
            self.accum[k] = {'inter_sum':0, 'pred_sum':0, 'out_sum':0, 'all':0}
            self.all_dices[k] = 0

        self.epsilon = epsilon
        self.mean_dice = 0

    

    def __call__(self, output, pred):
        output = output.argmax(dim = 1)
        bs = output.shape[0]
        for c in range(self.n_classes):
            cout_sum, cpred_sum, cint_sum = self.dice_values(self.get_mask(output, c), self.get_mask(pred, c))
            self.accum[self.classes[c]]['inter_sum'] += cint_sum
            self.accum[self.classes[c]]['pred_sum'] += cpred_sum
            self.accum[self.classes[c]]['out_sum'] += cout_sum

            self.all_dices[self.classes[c]] = self.dice(self.accum[self.classes[c]]['inter_sum'],
                                                        self.accum[self.classes[c]]['pred_sum'],
                                                        self.accum[self.classes[c]]['out_sum'])
            self.mean_dice = np.mean(list(self.all_dices.values()))

        self.n_count += bs

    def get_dice_scores(self):
        return self.all_dices

    def get_mean_dice_score(self, exeptions = []):
        a = []
        for k in self.classes:
            if k not in exeptions:
                a.append(self.all_dices[k])
        return np.mean(a)

    def dice(self, xy, x, y):
        return (2*xy + self.epsilon)/(x + y + self.epsilon)

    def dice_values(self, x,y):
        x_sum = x.sum().item()
        y_sum = y.sum().item()
        int_sum = (x*y).sum().item()
        return x_sum, y_sum, int_sum

    def get_mask(self, labels, i):
        return (labels == i)*1


# From https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
# Paper : http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf
# Modify to one hot encode the target
def dice_loss(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    x target is a 1-hot representation of the groundtruth, shoud have same size as the input x target is encoded as for CE pytorch loss
    """
    uniques=np.unique(target.cpu().numpy())
    n_classes = uniques.shape[0]
    target = _toEvaluationOneHot(target, n_classes)

    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs=torch.nn.functional.softmax(input)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
    

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=2*(num/(den1+den2))
    # print(dice.shape)
    # dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

    # dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
    dice_total = 1 - torch.sum(dice)/(n_classes*target.shape[0])
    return dice_total


def _toEvaluationOneHot(labels, n_classes):
    shape = labels.shape
    out = torch.zeros(*(shape[0], n_classes, shape[-2], shape[-1])).cuda()
    for i in range(n_classes):
        out[:,i, ...] = (labels == i)
    return out