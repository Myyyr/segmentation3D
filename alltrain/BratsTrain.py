from alltrain.Train import *
import  torch
from torch.utils.tensorboard import SummaryWriter
import time
import alltrain.bratsUtils as bratsUtils
from bratsDataset import *

from tqdm import tqdm

from torch.utils.data import DataLoader

class BTrain(Train):

    def __init__(self, expconfig):
        super(BTrain, self).__init__(expconfig)
        self.expconfig = expconfig

        self.device = torch.device("cuda")
        self.expconfig.net = expconfig.net.to(self.device)


        self.tb = SummaryWriter(comment=expconfig.experiment_name)

        self.bestMeanDice = 0
        self.bestMeanDiceEpoch = 0
        self.meanDice = 0
        self.smallmeanDice = 0

        trainDataset = BratsDataset(expconfig, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False)
        validDataset = BratsDataset(expconfig, mode="validation", randomCrop=None, hasMasks=True, returnOffsets=False)
        self.trainDataLoader = DataLoader(dataset=trainDataset, num_workers=4, batch_size=expconfig.batchsize, shuffle=True)
        self.valDataLoader = DataLoader(dataset=validDataset, num_workers=4, batch_size=expconfig.batchsize, shuffle=False)



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
            diceWT, diceTC, diceET = [], [], []
            sensWT, sensTC, sensET = [], [], []
            specWT, specTC, specET = [], [], []
            hdWT, hdTC, hdET = [], [], []

            smalldiceWT, smalldiceTC, smalldiceET = [], [], []
            smallsensWT, smallsensTC, smallsensET = [], [], []
            smallspecWT, smallspecTC, smallspecET = [], [], []
            smallhdWT, smallhdTC, smallhdET = [], [], []

            for i, data in tqdm(enumerate(self.valDataLoader), total = int(len(self.valDataLoader))):#enumerate(self.valDataLoader):
                inputs, _, labels, smalllabels = data
                inputs, labels, smalllabels = inputs.to(self.device), labels.to(self.device), smalllabels.to(self.device)
                outputs, smalloutputs = expcf.net(inputs)

                if expcf.train_original_classes:
                    outputsOriginal5 = outputs
                    outputs = torch.argmax(outputs, 1)
                    #hist, _ = np.histogram(outputs.cpu().numpy(), 5, (0, 4))
                    #buckets = buckets + hist
                    wt = bratsUtils.getWTMask(outputs)
                    tc = bratsUtils.getTCMask(outputs)
                    et = bratsUtils.getETMask(outputs)

                    smalloutputs = torch.argmax(smalloutputs, 1)
                    #hist, _ = np.histogram(smalloutputs.cpu().numpy(), 5, (0, 4))
                    #buckets = buckets + hist
                    wt = bratsUtils.getWTMask(smalloutputs)
                    tc = bratsUtils.getTCMask(smalloutputs)
                    et = bratsUtils.getETMask(smalloutputs)






                    labels = torch.argmax(labels, 1)
                    wtMask = bratsUtils.getWTMask(labels)
                    tcMask = bratsUtils.getTCMask(labels)
                    etMask = bratsUtils.getETMask(labels)

                    smalllabels = torch.argmax(smalllabels, 1)
                    smallwtMask = bratsUtils.getWTMask(smalllabels)
                    smalltcMask = bratsUtils.getTCMask(smalllabels)
                    smalletMask = bratsUtils.getETMask(smalllabels)

                else:

                    #separate outputs channelwise
                    wt, tc, et = outputs.chunk(3, dim=1)
                    s = wt.shape
                    wt = wt.view(s[0], s[2], s[3], s[4])
                    tc = tc.view(s[0], s[2], s[3], s[4])
                    et = et.view(s[0], s[2], s[3], s[4])

                    smallwt, smalltc, smallet = smalloutputs.chunk(3, dim=1)
                    s = smallwt.shape
                    smallwt = smallwt.view(s[0], s[2], s[3], s[4])
                    smalltc = smalltc.view(s[0], s[2], s[3], s[4])
                    smallet = smallet.view(s[0], s[2], s[3], s[4])





                    wtMask, tcMask, etMask = labels.chunk(3, dim=1)
                    s = wtMask.shape
                    wtMask = wtMask.view(s[0], s[2], s[3], s[4])
                    tcMask = tcMask.view(s[0], s[2], s[3], s[4])
                    etMask = etMask.view(s[0], s[2], s[3], s[4])

                    smallwtMask, smalltcMask, smalletMask = smalllabels.chunk(3, dim=1)
                    s = smallwtMask.shape
                    smallwtMask = smallwtMask.view(s[0], s[2], s[3], s[4])
                    smalltcMask = smalltcMask.view(s[0], s[2], s[3], s[4])
                    smalletMask = smalletMask.view(s[0], s[2], s[3], s[4])

                #TODO: add special evaluation metrics for original 5

                #get dice metrics
                diceWT.append(bratsUtils.dice(wt, wtMask))
                diceTC.append(bratsUtils.dice(tc, tcMask))
                diceET.append(bratsUtils.dice(et, etMask))

                # #get sensitivity metrics
                # sensWT.append(bratsUtils.sensitivity(wt, wtMask))
                # sensTC.append(bratsUtils.sensitivity(tc, tcMask))
                # sensET.append(bratsUtils.sensitivity(et, etMask))

                # #get specificity metrics
                # specWT.append(bratsUtils.specificity(wt, wtMask))
                # specTC.append(bratsUtils.specificity(tc, tcMask))
                # specET.append(bratsUtils.specificity(et, etMask))


                #get dice metrics
                smalldiceWT.append(bratsUtils.dice(smallwt, smallwtMask))
                smalldiceTC.append(bratsUtils.dice(smalltc, smalltcMask))
                smalldiceET.append(bratsUtils.dice(smallet, smalletMask))




                #calculate mean dice scores
            meanDiceWT = np.mean(diceWT)
            meanDiceTC = np.mean(diceTC)
            meanDiceET = np.mean(diceET)
            meanDice = np.mean([meanDiceWT, meanDiceTC, meanDiceET])
            self.meanDice = meanDice


            smallmeanDiceWT = np.mean(smalldiceWT)
            smallmeanDiceTC = np.mean(smalldiceTC)
            smallmeanDiceET = np.mean(smalldiceET)
            smallmeanDice = np.mean([smallmeanDiceWT, smallmeanDiceTC, smallmeanDiceET])
            self.smallmeanDice = smallmeanDice

            if (meanDice > self.bestMeanDice):
                self.bestMeanDice = meanDice
                self.bestMeanDiceEpoch = epoch

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