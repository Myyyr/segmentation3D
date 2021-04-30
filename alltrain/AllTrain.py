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

import alltrain.DiceScore as dc
import numpy as np

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
   #     self.valDataLoader = self.expconfig.valDataLoader
        self.testDataLoader = self.expconfig.testDataLoader

        self.save_dict = {'original':{} ,'small':{}}
        

        self.classes = self.expconfig.n_classes


    def step(self, expcf, inputs, labels, total_loss):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        # print('labels.shape :', labels.shape)
        # print(inputs.shape)
        outputs = expcf.net(inputs)
        del inputs
        loss = expcf.loss(outputs, labels)
        total_loss += loss.item()
        del outputs, labels
        return loss, total_loss

    def back_step(self, expcf, loss):
        loss.backward()
        if expcf.clip:
            torch.nn.utils.clip_grad_norm_(expcf.net.parameters(), 12)
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
        self.evaluate()
        exit(0)
        expcf = self.expconfig
        expcf.optimizer.zero_grad()
        print("#### EXPERIMENT : {} | ID : {} ####".format(expcf.experiment_name, expcf.id))
        print("#### TRAIN SET :", len(self.trainDataLoader))
   #     print("#### VALID SET :", len(self.valDataLoader))
        total_time = 0.0
        self.save_dict['first_batch_memory'] = ""
        min_loss = 1e10
        # self.evaluate()


        for epoch in range(expcf.start_epoch, expcf.epoch):
            startTime = time.time()
            expcf.net.train()


            total_loss = 0
            self.save_dict['epoch'] = epoch

            for i, data in tqdm(enumerate(self.trainDataLoader), total = int(len(self.trainDataLoader))) :
                #load data
                inputs, labels = data
                loss, total_loss = self.step(expcf, inputs, labels, total_loss)
                # self.tb.add_scalar("train_loss", loss.item(), epoch*int(len(self.trainDataLoader)) + i)
                del inputs, labels
                self.back_step(expcf, loss)
                del loss


            if self.save_dict['first_batch_memory'] == "":
                self.save_dict['first_batch_memory'] = self.convert_byte(torch.cuda.max_memory_allocated())

            print("epoch: {}, total_loss: {}, mem: {}".format(epoch, total_loss/int(len(self.trainDataLoader)), str(self.convert_byte(torch.cuda.max_memory_allocated())) ) )

            epochTime = time.time() - startTime
            total_time += epochTime
            


            #validation at end of epoch
            if epoch % expcf.validate_every_k_epochs == expcf.validate_every_k_epochs - 1:
                self.evaluate()

            #take lr sheudler step
            if expcf.lr_scheduler != None:
                expcf.lr_scheduler.step()

            # total_time += validTime
            # self.tb.add_scalar("totalTime", total_time, epoch)
            self.tb.add_scalar("lr", expcf.optimizer.param_groups[0]['lr'], epoch)
            self.tb.add_scalar("train_loss", total_loss/int(len(self.trainDataLoader)), epoch)
            # self.tb.add_scalar("ValidMeanDice", self.meanDice, epoch)
            # for k in self.expconfig.classes_name:
            #     self.tb.add_scalar(k+'_ValidDice', self.save_dict['original'][k], epoch)
            # self.tb.add_scalars('ValidClassesDice', self.save_dict['original'], epoch)
            
            
            # print("epoch: {}, meanDice: {}, memory : {}, Time : {}".format(epoch, 
            #                                                                 self.meanDice, 
            #                                                                 self.convert_byte(torch.cuda.max_memory_allocated()), 
            #                                                                 self.convert_time(total_time)) )
            print("epoch: {}, lr: {:.4f}, memory : {}, Time : {}".format(epoch, 
                                                            expcf.optimizer.param_groups[0]['lr'],
                                                            self.convert_byte(torch.cuda.max_memory_allocated()), 
                                                            self.convert_time(total_time)) )


            TL = total_loss
            if TL < min_loss:
                min_loss = TL
                self.saveToDisk(epoch)

        self.evaluate()
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


    def evaluate(self, vizonly=False):
        print("-"*20, "\nEVALUATION ...")
        expcf = self.expconfig

        with torch.no_grad():
            expcf.net.eval()

            dice = {}
            for pid in expcf.testDataset.used_pids:
                dice[str(pid)] = dc.DiceScore(self.expconfig.classes_name)

            for i, data in tqdm(enumerate(self.testDataLoader), total = int(len(self.testDataLoader))):
                pid, inputs, labels = data
                pid = int(pid[0,0].item())

                if not expcf.patched:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # print(i, 'inputs.shape', inputs.shape)
                    outputs = F.softmax(expcf.net(inputs), dim=1)
                    del inputs
                    dice[str(pid)](outputs, labels)
                    del labels, outputs
                else:
                    # print(inputs.shape)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # print(inputs.shape)
                    b, c, nh, nw, nd, h, w, d = inputs.shape

                    # print(labels.shape)
                    b, ah, aw, ad = labels.shape
                    outputs = torch.zeros((b, expcf.n_classes, ah, aw, ad))
                    # ps_w = int(w/nw)
                    # ps_h = int(h/nh)
                    # ps_d = int(d/nd)
                    crop = []
                    if expcf.testDataset.return_full_image:
                        for x in range(nh):
                            for y in range(nw):
                                for z in range(nd):
                                    crop.append(inputs[:,:,x,y,z,...])
                        crop = torch.cat(crop, dim=1)
                        # print(crop.shape)
                    # print(crop.shape)
                    
                    for x in range(nh):
                        for y in range(nw):
                            for z in range(nd):
                                if not expcf.testDataset.return_full_image:
                                    out_xyz = expcf.net(inputs[:,:,x,y,z,...])
                                    outputs[:, :, x*h:(x+1)*h, y*w:(y+1)*w, z*d:(z+1)*d] = out_xyz
                                else:
                                    inptc = inputs[:,:,x,y,z,...]
                                    # inputs = torch.reshape(inputs, (b,c,nh*nw*nd,h,w,d))
                                    print(crop.shape, inptc.shape)  
                                    print(torch.cat([inptc, crop]).shape)
                                    out_xyz = expcf.net(torch.cat([inptc, crop], 1))
                                    outputs[:, :, x*h:(x+1)*h, y*w:(y+1)*w, z*d:(z+1)*d] = out_xyz
                    if vizonly:
                        np.save('./viz_pred.npy', F.softmax(outputs, dim=1).cpu().numpy())
                        np.save('./viz_inputs.npy', inputs.cpu().numpy())
                        np.save('./viz_target.npy', labels.cpu().numpy())
                        exit(0)

                    dice[str(pid)](F.softmax(outputs, dim=1).detach().cuda(), labels)
            dices = {}
            classes_dices = {}
            for i in range(self.classes):
                classes_dices[self.expconfig.classes_name[i]] = []

            for pid in expcf.testDataset.used_pids:
                dices[str(pid)] = {}
                for i in range(self.classes):
                    dices[str(pid)][self.expconfig.classes_name[i]] = dice[str(pid)].get_dice_scores()[self.expconfig.classes_name[i]]
                    classes_dices[self.expconfig.classes_name[i]].append(dice[str(pid)].get_dice_scores()[self.expconfig.classes_name[i]])

                dices[str(pid)]["mean_over_orgs"] = dice[str(pid)].get_mean_dice_score(exeptions = ['background'])
            dices['means'] = {}
            for i in range(self.classes):
                dices['means'][self.expconfig.classes_name[i]] = np.mean(classes_dices[self.expconfig.classes_name[i]])
            dices['means']['mean_overall_orgs'] = np.mean(list(dices['means'].values()))

        print(dices['means'])
        with open(os.path.join(self.expconfig.checkpointsBasePath, self.expconfig.experiment_name+'_split_'+str(self.split)+'_evaluation_'+'.json'), 'w') as f:
            json.dump(dices, f, indent=4)


    # def evaluate3D(self):
    #     print("-"*20, "\nEVALUATION ...")
    #     expcf = self.expconfig

    #     with torch.no_grad():
    #         expcf.net.eval()

    #         dice = {}
    #         for pid in expcf.testDataset.data_splits:
    #             dice[str(pid)] = dc.DiceScore(self.expconfig.classes_name)

    #         for i, data in tqdm(enumerate(self.testDataLoader), total = int(len(self.testDataLoader))):
    #             pid, inputs, labels = data
    #             pid = int(pid[0,0].item())
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)

    #             outputs = F.softmax(expcf.net(inputs), dim=1)
    #             del inputs
    #             dice[str(pid)](outputs, labels)
    #             del labels, outputs

    #         dices = {}
    #         classes_dices = {}
    #         for i in range(self.classes):
    #             classes_dices[self.expconfig.classes_name[i]] = []

    #         for pid in expcf.testDataset.data_splits:
    #             dices[str(pid)] = {}
    #             for i in range(self.classes):
    #                 dices[str(pid)][self.expconfig.classes_name[i]] = dice[str(pid)].get_dice_scores()[self.expconfig.classes_name[i]]
    #                 classes_dices[self.expconfig.classes_name[i]].append(dice[str(pid)].get_dice_scores()[self.expconfig.classes_name[i]])

    #             dices[str(pid)]["mean_over_orgs"] = dice[str(pid)].get_mean_dice_score(exeptions = ['background'])
    #         dices['means'] = {}
    #         for i in range(self.classes):
    #             dices['means'][self.expconfig.classes_name[i]] = np.mean(classes_dices[self.expconfig.classes_name[i]])
    #         dices['means']['mean_overall_orgs'] = np.mean(list(dices['means'].values()))

    #     print(dices)
    #     with open(os.path.join(self.expconfig.checkpointsBasePath, self.expconfig.experiment_name+'_split_'+str(self.split)+'_evaluation_'+'.json'), 'w') as f:
    #         json.dump(dices, f, indent=4)





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