import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import numpy as np
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import Main_Module
from gen_imageset import DatasetGenerator,download_bsd300
import re, glob
import h5py_dataset

pretrained = 1
NUM_EPOCHS = 300



if torch.cuda.is_available():
    device = torch.device("cuda:0")
else :
    device = torch.device("cpu:0")

if __name__ == "__main__":

    cudnn.benchmark = True
    Batch_Size = 128
    NUM_WORKERS = 0

    dirpath = download_bsd300()
#    print(dirpath)

#    trainset = DatasetGenerator(dirpath,scaling_factor=2)
   # testset = DatasetGenerator(dirpath,scaling_factor=2)
    trainset = h5py_dataset.Read_dataset_h5("dataset/91-image_x2.h5")
    testset = h5py_dataset.Read_dataset_h5_Test("dataset/Set5_x2.h5")
    #print(testset.__len__())

    trainloader= DataLoader(dataset=trainset,shuffle=True,batch_size=Batch_Size,num_workers=NUM_WORKERS, drop_last=True)
    testloader = DataLoader(dataset=testset, batch_size=1, num_workers=NUM_WORKERS, drop_last=True)

    model = Main_Module.SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([{'params': model.conv1.parameters()},
                            {'params': model.conv2.parameters()},
                            {'params': model.conv3.parameters(), 'lr': 1e-5}], lr=1e-4)

    criterion = criterion.to(device)
    Model = Main_Module.SRCNN().to(device)


   # optimizer = optim.Adam(Model.parameters(), lr= 1e-4)

    psnr_array = np.zeros((NUM_EPOCHS))
    for epoch_ in range(NUM_EPOCHS):
        epoch_loss =0
        for i, batch in enumerate(trainloader):
            input, target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
           # print(np.array(input.cpu()).shape)
      #      print(np.array(input.cpu()).shape)
            out = model(input)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss +=loss.item()
           # print("training")

        print("Epoch {}. Training loss: {}".format(epoch_,epoch_loss/len(trainloader)))

        #test
        print("test phase")
        tot_psnr =0
        with torch.no_grad():
            for i,batch in enumerate(testloader):
                input, target= batch[0].to(device), batch[1].to(device)
                #print(np.array(input.cpu()).shape)
                out = model(input)
                loss = criterion(out, target)
                psnr = 10* np.log10(1/loss.item())
                tot_psnr += psnr
                print("ith PSNR : {}".format(psnr))
           # print(len(testloader))
            avg_psnr = tot_psnr/len(testloader)
            psnr_array[epoch_] = avg_psnr
            print("Average PSNR : {} dB.".format(avg_psnr))
            np.save("./psnr_array.npy",avg_psnr)

        torch.save(model,os.path.join("modeldata","model_{}_epoch.pth".format(epoch_)))
