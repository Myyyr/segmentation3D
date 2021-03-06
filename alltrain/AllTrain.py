from alltrain.Train import *
import  torch
from torch.utils.tensorboard import SummaryWriter
import time
# import alltrain.bratsUtils as bratsUtils
import alltrain.atlasUtils as atlasUtils
# from multiatlasDataset import *

from tqdm import tqdm
import json
import os
import numpy as np
import torch.nn.functional as F

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

class AllTrain(Train):

    def __init__(self, expconfig, split = 0):
        super(AllTrain, self).__init__(expconfig)
        self.expconfig = expconfig
        self.split = split
        self.startingTime = time.time()
        self.device = torch.device("cuda")
        # if self.expconfig.start_epoch == 0:
        self.expconfig.net = expconfig.net.to(self.device)
        optimizer_to(self.expconfig.optimizer, self.device)
        torch.cuda.empty_cache()


        self.tb = SummaryWriter(comment=expconfig.experiment_name)

        self.bestMeanDice = 0
        self.bestMeanDiceEpoch = 0
        self.meanDice = 0
        self.smallmeanDice = 0

        self.expconfig.set_data(split)

        self.trainDataLoader = self.expconfig.trainDataLoader
        self.valDataLoader = self.expconfig.valDataLoader

        self.save_dict = {'original':{} ,'small':{}}
        

        self.classes = self.expconfig.n_classes


    def step(self, expcf, inputs, labels, total_loss):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        # print('labels.shape :', labels.shape)
        outputs = expcf.net(inputs)
        del inputs
 
        loss = expcf.loss(outputs, labels)
        total_loss += loss.item()
        del outputs, labels
        return loss, total_loss

    def back_step(self, expcf, loss):
        loss.backward()

        #update params
        expcf.optimizer.step()
        if expcf.debug:
            L1, L2, L3 = [],[],[]
            for l in expcf.net.modules():
                if type(l) == torch.nn.Conv3d:
                    #L.append((l.weight.grad).mean().item())
                    # L1.append(((l.weight).mean().item()))
                    # L2.append(((l.weight).min().item()))
                    # L3.append(((l.weight).max().item()))
                    L1.append(torch.norm(l.weight.grad).item())
            print('mean :', L1)
            # print('min :', L2)
            # print('max :', L3)
        expcf.optimizer.zero_grad()
        del loss

    def train(self):
        expcf = self.expconfig
        expcf.optimizer.zero_grad()
        print("#### EXPERIMENT : {} | ID : {} ####".format(expcf.experiment_name, expcf.id))
        print("#### TRAIN SET :", len(self.trainDataLoader))
        print("#### VALID SET :", len(self.valDataLoader))
        total_time = 0.0
        self.save_dict['first_batch_memory'] = ""
        min_loss = 1e10

        for epoch in range(expcf.start_epoch, expcf.epoch):
            startTime = time.time()
            expcf.net.train()


            total_loss = 0
            self.save_dict['epoch'] = epoch

            for i, data in tqdm(enumerate(self.trainDataLoader), total = int(len(self.trainDataLoader))) :

                # expcf.net_stats()

                #load data
                inputs, labels = data

                
                
                loss, total_loss = self.step(expcf, inputs, labels, total_loss)
                del inputs, labels
                self.back_step(expcf, loss)
                del loss
                # torch.cuda.empty_cache()

            if self.save_dict['first_batch_memory'] == "":
                self.save_dict['first_batch_memory'] = str(self.convert_byte(torch.cuda.max_memory_allocated()))

            print("epoch: {}, total_loss: {}, mem: {}".format(epoch, total_loss/int(len(self.trainDataLoader)), str(self.convert_byte(torch.cuda.max_memory_allocated())) ) )

            epochTime = time.time() - startTime
            total_time += epochTime
            


            #validation at end of epoch
            if epoch % expcf.validate_every_k_epochs == expcf.validate_every_k_epochs - 1:
                validTime = self.validate(epoch)

            #take lr sheudler step
            if expcf.lr_scheduler != None:
                expcf.lr_scheduler.step()

            total_time += validTime
            self.tb.add_scalar("totalTime", total_time, epoch)

            self.tb.add_scalar("train_loss", total_loss/int(len(self.trainDataLoader)), epoch)

            self.tb.add_scalar("meanDice", self.meanDice, epoch)
            if expcf.look_small:
                self.tb.add_scalar("smallmeanDice", self.smallmeanDice, epoch)

            if expcf.look_small:
                print("epoch: {}, meanDice: {}, smallMeanDice: {}".format(epoch, self.meanDice, self.smallmeanDice))
            else:
                print("epoch: {}, meanDice: {}, memory : {}, Time : {}".format(epoch, 
                                                                            self.meanDice, 
                                                                            self.convert_byte(torch.cuda.max_memory_allocated()), 
                                                                            self.convert_time(total_time)) )


            TL = total_loss
            if TL < min_loss:
                min_loss = TL
                self.saveToDisk(epoch)

        self.tb.close()

    def convert_byte(self, v):
        units = {'Bytes':1,'KB':1e-3, 'MB':1e-6, 'GB':1e-9}
        tmp = 'Bytes'
        for k in list(units.keys()):
            if int(v*units[k]) == 0:
                return v*units[tmp], tmp
            tmp = k
        return v*units[tmp], tmp

    def convert_time(self, t):
        units = {'d':3600*24, 'h':3600, 'm':60, 's':1  }
        ret = ''
        for k in list(units.keys()):
            ret += str(int(t//units[k])  )+k
        return ret

    def valide_step(self, expcf, outputs, labels, dice, smalldice = None, smalllabels = None, smalloutputs = None):
        # print("before armax : out {}, lab {}".format(outputs.shape, labels.shape))
        outputs = outputs.argmax(dim = 1)    
        masks = []
        if labels.shape[1] == self.classes:
            labels = labels.argmax(dim = 1)
        label_masks = []
        # print("after armax : out {}, lab {}".format(outputs.shape, labels.shape))
        # a = 
        # print("unique : out {}, lab {}".format(np.unique(outputs.detach().cpu().numpy(),np.unique(labels.detach().cpu().numpy()))))

        # print('label :', np.unique(labels.cpu().numpy()))
        # print('outpu :', np.unique(outputs.cpu().numpy()))

        for i in range(self.classes):
            btch_dice = []
            for b in range(labels.shape[0]):
                mask = atlasUtils.getMask(outputs[b,...][None,...], i)
                label_mask = atlasUtils.getMask(labels[b,...][None,...], i)
                btch_dice.append(atlasUtils.dice(mask, label_mask))

            # print(" | mask {}, label_mask {}".format(mask.shape, label_mask.shape))
            # print(" | unique : mask {}, labmask {}".format(np.unique(mask.detach().cpu().numpy(),np.unique(label_mask.detach().cpu().numpy()))))

            # dice[i].append(atlasUtils.dice(mask, label_mask))
            dice[i].append(np.mean(btch_dice))
            del mask, label_mask
        del outputs, labels, label_masks, masks

        
    def validate(self, epoch):
        expcf = self.expconfig
        startTime = time.time()


        with torch.no_grad():
            expcf.net.eval()
            dice = [[] for i in range(self.classes)]
            for i, data in tqdm(enumerate(self.valDataLoader), total = int(len(self.valDataLoader))):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = F.softmax(expcf.net(inputs), dim=1)
                del inputs
                
                self.valide_step(expcf, outputs, labels, dice)
                del labels, outputs
            # print(dice)
            # print(len(dice))
            meanDices = []
            for i in range(self.classes):
                meanDices.append(np.mean(dice[i]))
                # print(meanDices)
                self.save_dict['original'][self.expconfig.classes_name[i]] = meanDices[i]
            print()
            self.meanDice = np.mean(meanDices)
            self.save_dict['meanDice'] =  self.meanDice 

            self.save_dict['epoch'] = epoch
            self.save_dict['memory'] = str(self.convert_byte(torch.cuda.max_memory_allocated()))
            self.save_dict['training_time'] =  time.time() - self.startingTime

            print(self.save_dict)

        self.save_results()

        return time.time() - startTime





    def saveToDisk(self, epoch):

        print("SAVE MODEL ...")

        #gather things to save
        saveDict = {"net_state_dict": self.expconfig.net.state_dict(),
                    "optimizer_state_dict": self.expconfig.optimizer.state_dict(),
                    "epoch": epoch
                    }
        if self.expconfig.lr_scheduler != None:
            saveDict["lr_scheduler_state_dict"] = self.expconfig.lr_scheduler.state_dict()

        #save dict
        basePath = self.expconfig.checkpointsBasePathMod + "{}".format(self.expconfig.id)
        path = basePath + "/mod.pt".format(epoch)
        

        if not os.path.exists(basePath):
            os.makedirs(basePath)
        torch.save(saveDict, path)

        with open(os.path.join(basePath, self.expconfig.experiment_name+'_split_'+str(self.split)+'.json'), 'w') as f:
            json.dump(self.save_dict, f)



    def save_results(self):
        with open(os.path.join(self.expconfig.checkpointsBasePath, self.expconfig.experiment_name+'_split_'+str(self.split)+'.json'), 'w') as f:
            json.dump(self.save_dict, f)

    def save_pred(self, x, y, py):
        np.save(os.path.join(self.expconfig.checkpointsBasePath, 'x.npy'), x)
        np.save(os.path.join(self.expconfig.checkpointsBasePath, 'y.npy'), y)
        np.save(os.path.join(self.expconfig.checkpointsBasePath, 'py.npy'), py)