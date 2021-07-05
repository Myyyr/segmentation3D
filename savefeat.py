from debug_conf import ExpConfig
from tqdm import tqdm
import numpy as np
import torch

savepath = "/local/DEEPLEARNING/MULTI_ATLAS/MULTI_ATLAS/FEATURES_AND_SKIPS/"


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
    hwd = [(512,12,12,3), (256,24,24,6), (128,48,48,12), (64,96,96,24), (32,192,192,48)]


    outputs = [torch.zeros((b, hwd[i][0], hwd[i][1]*nh, hwd[i][2]*nw, hwd[i][3]*nd)).float().cuda() for i in range(5)]
    crop = []


    for x in range(nh):
        for y in range(nw):
            for z in range(nd):
                in_pos = [torch.from_numpy(np.array((x,y,z)))[None, None, ...]]
                in_pos = torch.cat(in_pos+[pos], dim=1)
                out_xyz, S = exp.net(inputs[:,:,x,y,z,...], in_pos, True, True)
                outputs[0][:, :, x*hwd[0][1]:(x+1)*hwd[0][1], y*hwd[0][2]:(y+1)*hwd[0][2], z*hwd[0][3]:(z+1)*hwd[0][3]] = out_xyz
                outputs[1][:, :, x*hwd[1][1]:(x+1)*hwd[1][1], y*hwd[1][2]:(y+1)*hwd[1][2], z*hwd[1][3]:(z+1)*hwd[1][3]] = S[3]
                outputs[2][:, :, x*hwd[2][1]:(x+1)*hwd[2][1], y*hwd[2][2]:(y+1)*hwd[2][2], z*hwd[2][3]:(z+1)*hwd[2][3]] = S[2]
                outputs[3][:, :, x*hwd[3][1]:(x+1)*hwd[3][1], y*hwd[3][2]:(y+1)*hwd[3][2], z*hwd[3][3]:(z+1)*hwd[3][3]] = S[1]
                outputs[4][:, :, x*hwd[4][1]:(x+1)*hwd[4][1], y*hwd[4][2]:(y+1)*hwd[4][2], z*hwd[4][3]:(z+1)*hwd[4][3]] = S[0]



    for o in range(5):
        out = outputs[o].cpu().numpy()
        np.save(savepath+str(pid)+"_"+str(o)+".npy", out)



# Train
for i, data in tqdm(enumerate(trainDataLoader), total = int(len(trainDataLoader))):
    pid, pos, inputs, labels = data
    # print("INPUT -----> OK")
    pid = int(pid[0,0].item())

    inputs, labels = inputs.to(device), labels.to(device)
    # print("DEVICE -----> OK")
    b, c, nh, nw, nd, h, w, d = inputs.shape
    b, ah, aw, ad = labels.shape
    h, w, d = 12, 12, 3
    

    outputs = torch.zeros((b, 512, h*nh, w*nw, d*nd)).float().cuda()


    for x in range(nh):
        for y in range(nw):
            for z in range(nd):
                in_pos = [torch.from_numpy(np.array((x,y,z)))[None, None, ...]]
                in_pos = torch.cat(in_pos+[pos], dim=1)
                out_xyz = exp.net(inputs[:,:,x,y,z,...], in_pos, True, True)
                outputs[:, :, x*h:(x+1)*h, y*w:(y+1)*w, z*d:(z+1)*d] = out_xyz

    # print("FORWARD ALL -----> OK")
    outputs = outputs.cpu().numpy()
    # print("NUMPY -----> OK")
    np.save(savepath+str(pid)+".npy", outputs)
    # print("SAVING -----> OK")

    del outputs