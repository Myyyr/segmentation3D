from alltrain.Train import *
import  torch
from torch.utils.tensorboard import SummaryWriter
import time
# import alltrain.bratsUtils as bratsUtils
import alltrain.atlasUtils as atlasUtils
# from multiatlasDataset import *
from pancreasCTDataset import SplitTCIA3DDataset

from tqdm import tqdm

from torch.utils.data import DataLoader

import json
import os

class TCIA(Train):

    def __init__(self, expconfig, split = 0):
        super(TCIA, self).__init__(expconfig)
        self.expconfig = expconfig
        self.startingTime = time.time()

        self.device = torch.device("cuda")
        self.expconfig.net = expconfig.net.to(self.device)


        self.tb = SummaryWriter(comment=expconfig.experiment_name)

        self.bestMeanDice = 0
        self.bestMeanDiceEpoch = 0
        self.meanDice = 0
        self.smallmeanDice = 0

        # trainDataset = MultiAtlasDataset(expconfig, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False, split = split)
        # validDataset = MultiAtlasDataset(expconfig, mode="validation", randomCrop=None, hasMasks=True, returnOffsets=False, split = split)
        # self.trainDataLoader = DataLoader(dataset=trainDataset, num_workers=1, batch_size=expconfig.batchsize, shuffle=True)
        # self.valDataLoader = DataLoader(dataset=validDataset, num_workers=1, batch_size=expconfig.batchsize, shuffle=False)


        train_dataset = SplitTCIA3DDataset(ds_path, split='train', data_splits = data_splits['train'], im_dim=train_opts.im_dim, transform=ds_transform['train'], preload_data=train_opts.preloadData)
        test_dataset  = SplitTCIA3DDataset(ds_path, split='test',  data_splits = data_splits['test'],  im_dim=train_opts.im_dim, transform=ds_transform['valid'], preload_data=train_opts.preloadData)
        train_loader = DataLoader(dataset=train_dataset, num_workers=2, batch_size=train_opts.batchSize, shuffle=True)
        test_loader  = DataLoader(dataset=test_dataset,  num_workers=2, batch_size=train_opts.batchSize, shuffle=False)

        self.save_dict = {'original':{} ,'small':{}}
        self.split = split

        self.classes = 14


    def step(self, expcf, inputs, labels, total_loss):
        # print(labels.sum().item(), np.prod(labels.shape))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # expcf.net

        #forward and backward pass
        # a = 
        # print(a.sum().cpu().item(), np.prod(a.shape))
        # outputs = expcf.net.apply_argmax_softmax(expcf.net(inputs))
        outputs = expcf.net(inputs)
        # print(outputs.sum().cpu().item(), np.prod(outputs.shape))
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
                    L1.append(((l.weight).mean().item()))
                    L2.append(((l.weight).min().item()))
                    L3.append(((l.weight).max().item()))
            print('mean :', L1)
            print('min :', L2)
            print('max :', L3)
        expcf.optimizer.zero_grad()
        del loss

    def train(self):
        expcf = self.expconfig
        expcf.optimizer.zero_grad()
        print("#### EXPERIMENT : {} | ID : {} ####".format(expcf.experiment_name, expcf.id))
        print("#### TRAIN SET :", len(self.trainDataLoader))
        print("#### VALID SET :", len(self.valDataLoader))
        total_time = 0.0
        # self.validate(0)
        # exit(0)
        self.save_dict['first_batch_memory'] = ""

        for epoch in range(expcf.epoch):
            startTime = time.time()
            expcf.net.train()


            total_loss = 0

            for i, data in tqdm(enumerate(self.trainDataLoader), total = int(len(self.trainDataLoader))) :

                # expcf.net_stats()

                #load data
                if expcf.look_small:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data

                
                
                loss, total_loss = self.step(expcf, inputs, labels, total_loss)
                del inputs, labels
                self.back_step(expcf, loss)
                del loss

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
        # outputs = torch.argmax(outputs.cpu(), 1).short().to(self.device)
        # outputs = torch.argmax(outputs.half(), 1).short()
        outputs = outputs.argmax(dim = 1).short()
        # print('out unique',np.unique(outputs.cpu().numpy()))
        
        masks, smallmasks = [], []


        # labels = torch.argmax(labels.cpu(), 1).short().to(self.device)
        # labels = torch.argmax(labels, 1).short()
        labels = labels.argmax(dim = 1).short()
        # print('lab unique',np.unique(labels.cpu().numpy()))

        if expcf.look_small:
            smalllabels = torch.argmax(smalllabels, 1)
        label_masks, smalllabel_masks = [], []


        for i in range(self.classes):
            mask = atlasUtils.getMask(outputs, i)
            label_mask = atlasUtils.getMask(labels, i)
            dice.append(atlasUtils.dice(mask, label_mask))
            del mask, label_mask

            if expcf.look_small:
                smallmasks.append(atlasUtils.getMask(smalloutputs, i))                        
                smalllabel_masks.append(atlasUtils.getMask(smalllabels, i))
                smalldice.append(atlasUtils.dice(smallmasks[i], smalllabel_masks[i]))

        del outputs, labels, label_masks, masks
        if expcf.look_small:
            del smalloutputs, smalllabels, smallmasks, smalllabel_masks
 
    def validate(self, epoch):
        expcf = self.expconfig
        
        startTime = time.time()


        with torch.no_grad():
            expcf.net.eval()
            dice = []
            smalldice = []

            for i, data in tqdm(enumerate(self.valDataLoader), total = int(len(self.valDataLoader))):#enumerate(self.valDataLoader):
               
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = expcf.net(inputs)
                # self.save_pred(inputs.cpu().numpy(), labels.cpu().numpy(), outputs.cpu().numpy())
                smalldice, smalllabels, smalloutputs = None, None, None
                del inputs
                
                self.valide_step(expcf, outputs, labels, dice, smalldice = smalldice, smalllabels = smalllabels, smalloutputs = smalloutputs)
                del labels, outputs
                
             

            meanDices, smallmeanDices = [], []
            for i in range(self.classes):
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

        return time.time() - startTime





    def saveToDisk(self, epoch):

        #gather things to save
        saveDict = {"net_state_dict": self.expconfig.net.state_dict(),
                    "optimizer_state_dict": self.expconfig.optimizer.state_dict(),
                    "epoch": epoch
                    }
        if self.expconfig.lr_scheduler != None:
            saveDict["lr_scheduler_state_dict"] = self.expconfig.lr_scheduler.state_dict()

        #save dict
        basePath = self.expconfig.checkpointsBasePathMod + "{}".format(self.expconfig.id)
        path = basePath + "/e_{}.pt".format(epoch)
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        torch.save(saveDict, path)


    def save_results(self):
        with open(os.path.join(self.expconfig.checkpointsBasePath, self.expconfig.experiment_name+'_split_'+str(self.split)+'.json'), 'w') as f:
            json.dump(self.save_dict, f)

    def save_pred(self, x, y, py):
        np.save(os.path.join(self.expconfig.checkpointsBasePath, 'x.npy'), x)
        np.save(os.path.join(self.expconfig.checkpointsBasePath, 'y.npy'), y)
        np.save(os.path.join(self.expconfig.checkpointsBasePath, 'py.npy'), py)