from alltrain.Train import *
import  torch
from torch.utils.tensorboard import SummaryWriter
import time
# import alltrain.bratsUtils as bratsUtils
import alltrain.atlasUtils as atlasUtils
from multiatlasDataset import *

from tqdm import tqdm

from torch.utils.data import DataLoader

import json
import os
class MemMATrain(Train):

    def __init__(self, expconfig, split = 0):
        super(MemMATrain, self).__init__(expconfig)
        self.expconfig = expconfig
        self.startingTime = time.time()

        self.device = torch.device("cuda")
        self.expconfig.net = expconfig.net.to(self.device)


        self.tb = SummaryWriter(comment=expconfig.experiment_name)

        self.bestMeanDice = 0
        self.bestMeanDiceEpoch = 0
        self.meanDice = 0
        self.smallmeanDice = 0

        trainDataset = MultiAtlasDataset(expconfig, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False, split = split)
        validDataset = MultiAtlasDataset(expconfig, mode="validation", randomCrop=None, hasMasks=True, returnOffsets=False, split = split)
        self.trainDataLoader = DataLoader(dataset=trainDataset, num_workers=1, batch_size=expconfig.batchsize, shuffle=True)
        self.valDataLoader = DataLoader(dataset=validDataset, num_workers=1, batch_size=expconfig.batchsize, shuffle=False)

        self.save_dict = {'original':{} ,'small':{}}
        self.split = split


    def prt_mem(self, txt, flt_type = 4):
        a = torch.cuda.max_memory_allocated()
        b = torch.cuda.memory_allocated()
        print(txt,':' ,self.convert_byte(a) , self.convert_byte(b))

    def step(self, expcf, inputs, labels, total_loss):
        inputs = inputs.to(self.device)
        self.prt_mem('inputs')
        labels = labels.to(self.device)
        self.prt_mem('labels')
        # expcf.net

        #forward and backward pass
        # outputs= expcf.net.apply_argmax_softmax(expcf.net(inputs))
        outputs = expcf.net(inputs)
        self.prt_mem('forward')

        loss = expcf.loss(outputs, labels)
        self.prt_mem('loss')
        total_loss += loss.item()
        del inputs, outputs, labels
        self.prt_mem('del in, out, lab')
        return loss, total_loss

    def back_step(self, expcf, loss):
        loss.backward()
        self.prt_mem('backward')

        #update params
        expcf.optimizer.step()
        self.prt_mem('opti step')
        expcf.optimizer.zero_grad()
        self.prt_mem('zero')
        del loss
        self.prt_mem('del loss')

    def train(self):
        expcf = self.expconfig
        expcf.optimizer.zero_grad()
        print("#### EXPERIMENT : {} | ID : {} ####".format(expcf.experiment_name, expcf.id))
        print("#### TRAIN SET :", len(self.trainDataLoader))
        print("#### VALID SET :", len(self.valDataLoader))
        total_time = 0.0
        # self.validate(0)
        # exit(0)

        self.prt_mem('start')

        for epoch in range(expcf.epoch):
            startTime = time.time()
            expcf.net.train()


            total_loss = 0

            for i, data in tqdm(enumerate(self.trainDataLoader), total = int(len(self.trainDataLoader))) :

                

                #load data
                if expcf.look_small:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data
                
                self.prt_mem('before_step')
                loss, total_loss = self.step(expcf, inputs, labels, total_loss)
                self.prt_mem('after step')
                del inputs, labels
                self.prt_mem('after del in out')
                self.back_step(expcf, loss)
                self.prt_mem('after back_step')
                del loss
                self.prt_mem('after del loss')

                

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
            self.tb.add_scalar("smallmeanDice", self.smallmeanDice, epoch)

            print("epoch: {}, bestMeanDice: {}, meanDice: {}, smallMeanDice: {}".format(epoch, self.bestMeanDice, self.meanDice, self.smallmeanDice))


            


        self.tb.close()

    def convert_byte(self, v):
        units = {'Bytes':1,'KB':1e-3, 'MB':1e-6, 'GB':1e-9}
        tmp = 'Bytes'
        for k in list(units.keys()):
            if int(v*units[k]) == 0:
                return v*units[tmp], tmp
            tmp = k
        return v*units[tmp], tmp


    def valide_step(self, expcf, outputs, labels, dice, smalldice = None, smalllabels = None, smalloutputs = None):
        outputs = torch.argmax(outputs.cpu(), 1).short().to(self.device)
        print('#### SHAPE :' ,outputs.shape)
        self.prt_mem('#argamax outputs')
        if expcf.look_small:
            smalloutputs = torch.argmax(smalloutputs, 1)

        masks, smallmasks = [], []


        labels = torch.argmax(labels.cpu(), 1).short().to(self.device)
        if expcf.look_small:
            smalllabels = torch.argmax(smalllabels, 1)
        label_masks, smalllabel_masks = [], []

        self.prt_mem('#argmax label')

        for i in range(12):
            mask = atlasUtils.getMask(outputs, i)
            label_mask = atlasUtils.getMask(labels, i)
            dice.append(atlasUtils.dice(mask, label_mask))

            if expcf.look_small:
                smallmasks.append(atlasUtils.getMask(smalloutputs, i))                        
                smalllabel_masks.append(atlasUtils.getMask(smalllabels, i))
                smalldice.append(atlasUtils.dice(smallmasks[i], smalllabel_masks[i]))

        self.prt_mem('#masks')    
        del outputs, labels, label_masks, masks
        if expcf.look_small:
            del smalloutputs, smalllabels, smallmasks, smalllabel_masks
        self.prt_mem('#del all')
 
    def validate(self, epoch):
        expcf = self.expconfig
        
        startTime = time.time()

        self.prt_mem('#VALIDATE start')

        with torch.no_grad():
            expcf.net.eval()
            dice = []
            smalldice = []

            for i, data in tqdm(enumerate(self.valDataLoader), total = int(len(self.valDataLoader))):#enumerate(self.valDataLoader):
                if expcf.look_small:
                    inputs, labels, smalllabels = data
                    inputs, labels, smalllabels = inputs.to(self.device), labels.to(self.device), smalllabels.to(self.device)
                    outputs, smalloutputs = expcf.net(inputs)
                    del inputs
                else:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.prt_mem('#input/out')
                    outputs= expcf.net(inputs)
                    self.prt_mem('#forward')
                    smalldice, smalllabels, smalloutputs = None, None, None
                    del inputs
                
                self.valide_step(expcf, outputs, labels, dice, smalldice = smalldice, smalllabels = smalllabels, smalloutputs = smalloutputs)
                self.prt_mem('#after_validate_step')
                del labels, outputs
                self.prt_mem('#del lab out')
                
             

            meanDices, smallmeanDices = [], []
            for i in range(12):
                meanDices.append(np.mean(dice[i]))
                self.save_dict['original'][self.expconfig.classes_name[i]] = meanDices[i]

                if expcf.look_small:
                    smallmeanDices.append(np.mean(smalldice[i]))
                    self.save_dict['small'][self.expconfig.classes_name[i]] = smallmeanDices[i]

            self.meanDice = np.mean([j for j in meanDices])
            self.save_dict['meanDice'] =  self.meanDice 

            if expcf.look_small:
                self.smallmeanDice = np.mean([j for j in smallmeanDices])
                self.save_dict['smallmeanDice'] =  self.smallmeanDice 

            self.save_dict['epoch'] = epoch
            self.save_dict['memory'] = str(self.convert_byte(torch.cuda.max_memory_allocated()))
            self.save_dict['training_time'] =  time.time() - self.startingTime



        self.save_results()

        self.prt_mem('#after valid')
        return time.time() - startTime





    def saveToDisk(self, epoch):

        #gather things to save
        saveDict = {"net_state_dict": self.expconfig.net.state_dict(),
                    "optimizer_state_dict": self.expconfig.optimizer.state_dict(),
                    "epoch": epoch,
                    "bestMeanDice": self.bestMeanDice,
                    "bestMeanDiceEpoch": self.bestMeanDiceEpoch
                    }
        if self.expconfig.lr_scheduler != None:
            saveDict["lr_scheduler_state_dict"] = self.expconfig.lr_scheduler.state_dict()

        #save dict
        basePath = self.expconfig.checkpointsBasePathSave + "{}".format(self.expconfig.id)
        path = basePath + "/e_{}.pt".format(epoch)
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        torch.save(saveDict, path)


    def save_results(self):
        with open(os.path.join(self.expconfig.checkpointsBasePath, self.expconfig.experiment_name+'_split_'+str(self.split)+'.json'), 'w') as f:
            json.dump(self.save_dict, f)