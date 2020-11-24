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
class MATrain(Train):

    def __init__(self, expconfig, split = 0):
        super(MATrain, self).__init__(expconfig)
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

    def train(self):
        expcf = self.expconfig
        expcf.optimizer.zero_grad()
        print("#### EXPERIMENT : {} | ID : {} ####".format(expcf.experiment_name, expcf.id))
        total_time = 0.0


        for epoch in range(expcf.epoch):
            startTime = time.time()
            expcf.net.train()

            total_loss = 0

            for i, data in tqdm(enumerate(self.trainDataLoader), total = int(len(self.trainDataLoader))) :

                

                #load data
                inputs, pid, labels, _ = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                #forward and backward pass
                outputs, _ = expcf.net(inputs)

                loss = expcf.loss(outputs, labels)
                total_loss += loss.item()
                del inputs, outputs, labels
                loss.backward()

                #update params
                expcf.optimizer.step()
                expcf.optimizer.zero_grad()


            print("epoch: {}, total_loss: {}, mem: {}".format(epoch, total_loss/int(len(self.trainDataLoader)), self.convert_bytes(torch.cuda.max_memory_allocated())))

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

    def convert_bytes(self, size):
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return "%3.2f %s" % (size, x)
            size /= 1024.0

        return size


    def validate(self, epoch):
        expcf = self.expconfig
        expcf.net.eval()
        startTime = time.time()

        with torch.no_grad():
            dice = []
            smalldice = []

            for i, data in tqdm(enumerate(self.valDataLoader), total = int(len(self.valDataLoader))):#enumerate(self.valDataLoader):
                if expcf.look_small:
                    inputs, _, labels, smalllabels = data
                    inputs, labels, smalllabels = inputs.to(self.device), labels.to(self.device), smalllabels.to(self.device)
                    outputs, smalloutputs = expcf.net(inputs)
                else:
                    inputs, _, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = expcf.net(inputs)

                
                outputs = torch.argmax(outputs, 1)
                if expcf.look_small:
                    smalloutputs = torch.argmax(smalloutputs, 1)

                masks, smallmasks = [], []


                labels = torch.argmax(labels, 1)
                smalllabels = torch.argmax(smalllabels, 1)
                label_masks, smalllabel_masks = [], []



                for i in range(12):
                    masks.append(atlasUtils.getMask(outputs, i))
                    label_masks.append(atlasUtils.getMask(labels, i))
                    dice.append(atlasUtils.dice(masks[i], label_masks[i]))

                    if expcf.look_small:
                        smallmasks.append(atlasUtils.getMask(smalloutputs, i))                        
                        smalllabel_masks.append(atlasUtils.getMask(smalllabels, i))
                        smalldice.append(atlasUtils.dice(smallmasks[i], smalllabel_masks[i]))

                    


             

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
            self.save_dict['memory'] = convert_bytes(torch.cuda.max_memory_allocated())
            self.save_dict['training_time'] =  time.time() - self.startingTime



        self.save_results()


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