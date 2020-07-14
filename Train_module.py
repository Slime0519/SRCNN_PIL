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

pretrained = 1
filepath = 'checkpoint/SRCNN_Adam_epoch_500.tar'
NUM_EPOCHS =200



if torch.cuda.is_available():
    device = torch.device("cuda:0")
else :
    device = torch.device("cpu:0")

if __name__ == "__main__":

    cudnn.benchmark = True
    Batch_Size = 4
    NUM_WORKERS = 0

    dirpath = download_bsd300()
    print(dirpath)

    trainset = DatasetGenerator(dirpath,scaling_factor=2)
    testset = DatasetGenerator(dirpath,scaling_factor=2)

    trainloader= DataLoader(dataset=trainset,shuffle=True,batch_size=Batch_Size,num_workers=NUM_WORKERS)
    testloader = DataLoader(dataset=testset, shuffle=True, batch_size=Batch_Size, num_workers=NUM_WORKERS)

    model = Main_Module.SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([{'params': model.conv1.parameters()},
                            {'params': model.conv2.parameters()},
                            {'params': model.conv3.parameters(), 'lr': 1e-5}], lr=1e-4)

    criterion = criterion.to(device)
    Model = Main_Module.SRCNN().to(device)


    optimizer = optim.Adam(Model.parameters(), lr= 1e-4)


    for epoch_ in range(NUM_EPOCHS):
        epoch_loss =0
        for i, batch in enumerate(trainloader):
            input, target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
           # print(np.array(input.cpu()).shape)
            out = model(input)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss +=loss.item()

        print("Epoch {}. Training loss: {}".format(epoch_,epoch_loss/len(trainloader)))

        #test
        avg_psnr =0
        with torch.no_grad():
            for batch in testloader:
                input, target= batch[0].to(device), batch[1].to(device)

                out = model(input)
                loss = criterion(input, target)
                psnr = 10* np.log10(1/loss.item())
                avg_psnr += psnr
        print("Average PSNR : {} dB.".format(avg_psnr/len(testloader)))

        torch.save(model,os.path.join("modeldata","model_{}_epoch.pth".format(epoch_)))