from Train import *
import  torch
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import alltrain.bratsUtils
from bratsDataset import *

from torch.utils.data import DataLoader

class BratsTrain(Train):

    def __init__(self, expconfig):
        super(BratsTrain, self).__init__()
        self.expconfig = expconfig

        self.device = torch.device("cuda")
        self.expconfig.net = expconfig.net.to(self.device)


        self.tb = SummaryWriter()

        self.bestMeanDice = 0
        self.bestMeanDiceEpoch = 0
        self.meanDice = 0

        trainDataset = BratsDataset(expConfig, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False)
        validDataset = BratsDataset(expConfig, mode="valid", randomCrop=None, hasMasks=True, returnOffsets=False)
        self.trainDataLoader = DataLoader(dataset=trainDataset, num_workers=4, batch_size=train_opts.batchSize, shuffle=True)
        self.valDataLoader = DataLoader(dataset=validDataset, num_workers=4, batch_size=train_opts.batchSize, shuffle=False)



    def train(self):
        expcf = self.expconfig
        expconfig.optimizer.zero_grad()
        print("#### EXPERIMENT : {} | ID : {} ####".format(expcf.experiment_name, expcf.id))
        total_time = 0.0


        for epoch in range(expcf.epoch):
            startTime = time.time()
            expcf.net.train()

            total_loss = 0

            for i, data in enumerate(self.trainDataLoader):

                #load data
                inputs, pid, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                #forward and backward pass
                outputs = expcf.net(inputs)
                loss = expcf.loss(outputs, labels)
                total_loss += loss.item()
                del inputs, outputs, labels
                loss.backward()

                #update params
                expcf.optimizer.step()
                expcf.optimizer.zero_grad()

            epochTime = time.time() - startTime
            total_time += epochTime
            self.tb.add_scalar("trainingTime", epochTime, epoch)
            


            #validation at end of epoch
            if epoch % expcf.validate_every_k_epochs == expcf.validate_every_k_epochs - 1:
                validTime = self.validate(epoch)

            #take lr sheudler step
            if expcf.lr_scheduler != None:
                expcf.lr_scheduler.step()

            total_time += validTime
            self.tb.add_scalar("validTime", validTime, epoch)
            self.tb.add_scalar("totalTime", total_time, epoch)

            self.tb.add_scalar("train_loss", total_loss, epoch)

            self.tb.add_scalar("bestMeanDice", self.bestMeanDice, epoch)
            self.tb.add_scalar("bestMeanDiceEpoch", self.bestMeanDiceEpoch, epoch)
            self.tb.add_scalar("meanDice", self.meanDice, epoch)


            


        self.tb.close()




    def validate(self, epoch):
        expcf = self.expconfig
        expcf.net.eval()
        startTime = time.time()

        with torch.no_grad():
            diceWT, diceTC, diceET = [], [], []
            sensWT, sensTC, sensET = [], [], []
            specWT, specTC, specET = [], [], []
            hdWT, hdTC, hdET = [], [], []

            for i, data in enumerate(self.valDataLoader):
                inputs, _, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = expConfig.net(inputs)

                if expConfig.TRAIN_ORIGINAL_CLASSES:
                    outputsOriginal5 = outputs
                    outputs = torch.argmax(outputs, 1)
                    #hist, _ = np.histogram(outputs.cpu().numpy(), 5, (0, 4))
                    #buckets = buckets + hist
                    wt = bratsUtils.getWTMask(outputs)
                    tc = bratsUtils.getTCMask(outputs)
                    et = bratsUtils.getETMask(outputs)

                    labels = torch.argmax(labels, 1)
                    wtMask = bratsUtils.getWTMask(labels)
                    tcMask = bratsUtils.getTCMask(labels)
                    etMask = bratsUtils.getETMask(labels)

                else:

                    #separate outputs channelwise
                    wt, tc, et = outputs.chunk(3, dim=1)
                    s = wt.shape
                    wt = wt.view(s[0], s[2], s[3], s[4])
                    tc = tc.view(s[0], s[2], s[3], s[4])
                    et = et.view(s[0], s[2], s[3], s[4])

                    wtMask, tcMask, etMask = labels.chunk(3, dim=1)
                    s = wtMask.shape
                    wtMask = wtMask.view(s[0], s[2], s[3], s[4])
                    tcMask = tcMask.view(s[0], s[2], s[3], s[4])
                    etMask = etMask.view(s[0], s[2], s[3], s[4])

                #TODO: add special evaluation metrics for original 5

                #get dice metrics
                diceWT.append(bratsUtils.dice(wt, wtMask))
                diceTC.append(bratsUtils.dice(tc, tcMask))
                diceET.append(bratsUtils.dice(et, etMask))

                #get sensitivity metrics
                sensWT.append(bratsUtils.sensitivity(wt, wtMask))
                sensTC.append(bratsUtils.sensitivity(tc, tcMask))
                sensET.append(bratsUtils.sensitivity(et, etMask))

                #get specificity metrics
                specWT.append(bratsUtils.specificity(wt, wtMask))
                specTC.append(bratsUtils.specificity(tc, tcMask))
                specET.append(bratsUtils.specificity(et, etMask))


                #calculate mean dice scores
            meanDiceWT = np.mean(diceWT)
            meanDiceTC = np.mean(diceTC)
            meanDiceET = np.mean(diceET)
            meanDice = np.mean([meanDiceWT, meanDiceTC, meanDiceET])
            self.meanDice = meanDice

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