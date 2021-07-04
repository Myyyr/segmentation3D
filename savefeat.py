from debug_conf import ExpConfig
from tqdm import tqdm
import numpy as np
import torch

savepath = "/local/DEEPLEARNING/MULTI_ATLAS/MULTI_ATLAS/FEATURES/"


exp = ExpConfig()
device = torch.device("cuda")

exp.net.float().to(device)

exp.set_data()

trainDataLoader = exp.trainDataLoader
testDataLoader = exp.testDataLoader


exp.net.eval()


# Test
for i, data in tqdm(enumerate(testDataLoader), total = int(len(testDataLoader))):
    pid, pos, inputs, labels = data
    pid = int(pid[0,0].item())

    inputs, labels = inputs.to(device), labels.to(device)
    b, c, nh, nw, nd, h, w, d = inputs.shape
    b, ah, aw, ad = labels.shape
    h, w, d = 12, 12, 3

    outputs = torch.zeros((b, 512, h*nh, w*nw, d*nd)).float().cuda()
    crop = []


    for x in range(nh):
        for y in range(nw):
            for z in range(nd):
                in_pos = [torch.from_numpy(np.array((x,y,z)))[None, None, ...]]
                in_pos = torch.cat(in_pos+[pos], dim=1)
                out_xyz = exp.net(inputs[:,:,x,y,z,...], in_pos, True, True)
                print(out_xyz.shape)
                outputs[:, :, x*h:(x+1)*h, y*w:(y+1)*w, z*d:(z+1)*d] = out_xyz


    outputs = outputs.cpu().numpy()
    np.save(savepath+"test/"+str(pid)+".npy", outputs)


# Train
for i, data in tqdm(enumerate(trainDataLoader), total = int(len(trainDataLoader))):
    pid, pos, inputs, labels = data
    pid = int(pid[0,0].item())

    inputs, labels = inputs.to(device), labels.to(device)
    b, c, nh, nw, nd, h, w, d = inputs.shape
    b, ah, aw, ad = labels.shape

    outputs = torch.zeros((b, 512, h*nh, w*nw, d*nd)).float().cuda()
    crop = []


    for x in range(nh):
        for y in range(nw):
            for z in range(nd):
                in_pos = [torch.from_numpy(np.array((x,y,z)))[None, None, ...]]
                in_pos = torch.cat(in_pos+[pos], dim=1)
                out_xyz = exp.net(inputs[:,:,x,y,z,...], in_pos, True, True)
                outputs[:, :, x*h:(x+1)*h, y*w:(y+1)*w, z*d:(z+1)*d] = out_xyz


    outputs = outputs.cpu().numpy()
    np.save(savepath+"train/"+str(pid)+".npy", outputs)