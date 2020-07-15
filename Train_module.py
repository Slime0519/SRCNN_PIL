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
import matplotlib.pyplot as plt

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

    eval_psnr_array = np.zeros((NUM_EPOCHS))
    train_psnr_array = np.zeros((NUM_EPOCHS))
    for epoch_ in range(NUM_EPOCHS):
        epoch_loss =0
        train_tot_psnr=0
        print("--------train phase--------")
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
            psnr = 10 * np.log10(1 / loss.item())
            train_tot_psnr += psnr
            """ print patch
            temp1 = np.squeeze(input[80].cpu().numpy())
            temp2 = np.squeeze(target[80].cpu().numpy())
            plt.imshow(temp1)
            plt.show()
            plt.imshow(temp2)
            plt.show()
            exit()
            """
           # print("Train : {}th PSNR : {}".format(i, psnr))
        # print("training")
        avg_psnr = train_tot_psnr / len(trainloader)
        train_psnr_array[epoch_] = avg_psnr
        print("Train : Average PSNR : {} dB.".format(avg_psnr))
        np.save("./train_psnr_array.npy", train_psnr_array)

        print("Epoch {}. Training loss: {}".format(epoch_,epoch_loss/len(trainloader)))

        #test
        print("--------test phase--------")
        eval_tot_psnr =0
        with torch.no_grad():
            for i,batch in enumerate(testloader):
                input, target= batch[0].to(device), batch[1].to(device)
                #print(np.array(input.cpu()).shape)
                out = model(input)
                loss = criterion(out, target)
                psnr = 10* np.log10(1/loss.item())
                eval_tot_psnr += psnr
                print("Eval : {}th PSNR : {}".format(i,psnr))

               #이미지 저장
                if (epoch_ == 0) & ((i==3) | (i==2)):
                    plt.imshow(np.squeeze(input[0].cpu().numpy()))
                    plt.title("original input")
                    plt.show()
                    plt.imshow(np.squeeze(target[0].cpu().numpy()))
                    plt.title("target image")
                    plt.show()
                    plt.imshow(np.squeeze(input[0].cpu().numpy()))
                    plt.title("original input")
                    plt.savefig("Imagedata/original_input_{}th_image".format(i + 1), dpi=500)
                    plt.imshow(np.squeeze(target[0].cpu().numpy()))
                    plt.title("target image")
                    plt.savefig("Imagedata/target_{}th_image".format(i + 1), dpi=500)
                    plt.imshow(np.squeeze(out[0].cpu().numpy()))
                    plt.title("epoch : {}, PSNR : {}".format(epoch_ + 1, psnr))
                    plt.show()
                    plt.imshow(np.squeeze(out[0].cpu().numpy()))
                    plt.title("epoch : {}, PSNR : {}".format(epoch_ + 1, psnr))
                    plt.savefig("Imagedata/{}th_image_{}th_epoch".format(i + 1, epoch_ + 1), dpi=500)
                if ((epoch_+1)% 50 == 0) & ((i==3) | (i==2)):
                    plt.imshow(np.squeeze(out[0].cpu().numpy()))
                    plt.title("epoch : {}, PSNR : {}".format(epoch_+1,psnr))
                    plt.show()
                    plt.imshow(np.squeeze(out[0].cpu().numpy()))
                    plt.title("epoch : {}, PSNR : {}".format(epoch_ + 1, psnr))
                    plt.savefig("Imagedata/{}th_image_{}th_epoch".format(i+1,epoch_+1),dpi = 500)

           # print(len(testloader))
            avg_psnr = eval_tot_psnr/len(testloader)
            eval_psnr_array[epoch_] = avg_psnr
            print("Eval Average PSNR : {} dB.".format(avg_psnr))
            np.save("./eval_psnr_array.npy",eval_psnr_array)

        torch.save(model,os.path.join("modeldata","model_{}_epoch.pth".format(epoch_)))
