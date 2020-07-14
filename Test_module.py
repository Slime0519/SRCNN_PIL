import torch
import h5py_dataset
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2

from torch.utils.data import DataLoader

import gen_imageset
import Main_Module
import glob
import os

testset_dir = "/Train"


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu:0')


def PSNR(img1,img2):
    MSE = (np.sum(img1-img2)**2)/(img1.shape[0]*img1.shape[1])
    PSNR = 10*np.log(1.0/MSE)
    return PSNR


if __name__=="__main__":
    #make SRCNN network
    SRCNN_Test = Main_Module.SRCNN()

    #checkpoint = torch.load(os.path.join("checkpoint","srcnn_x2.pth"))
    #SRCNN_Test.load_state_dict(checkpoint)
    SRCNN_Test = SRCNN_Test.to(device)
    print(SRCNN_Test)

    state_dict = SRCNN_Test.state_dict()
    for n, p in torch.load(os.path.join("checkpoint","srcnn_x2.pth")).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError


    SRCNN_Test.eval()

    #generate image list(original, distorted(input), labeled)
    filelist = glob.glob(os.path.join("Train","*.bmp"))
    print(filelist)


    image_list = []
    Test_label = []
    Test_input = []
    for filename in filelist:
        image = gen_imageset.load_img(filename)
        #plt.imshow(image)
       # plt.show()
        image_list.append(image)
        input, label = gen_imageset.rescaling_img(image[:,:,0], 2)
        #Test_label.append(image[:,:,0])
        #Test_input.append(gen_imageset.)
        Test_input.append(input)
        Test_label.append(label)

    image_list = np.array(image_list)
    Test_label = np.array(Test_label)
    Test_input = np.array(Test_input)

    Tot_PSNR =0
    output_list = []
    #print(len(Test_input))
    for i in range(len(Test_input)): #len(Test_input) = 91
        input = np.expand_dims(Test_input[i],(0,1))
        input = torch.from_numpy(input).float().to(device)
        output = SRCNN_Test(input)

        output_edit = output.cpu()
        output_edit = output_edit.detach().numpy()
        output_edit = output_edit.squeeze(0).squeeze(0)

        label = Test_label[i]
        Tot_PSNR += PSNR(output_edit,label[8:-8,8:-8] )

        output_list.append(output_edit)
        """
        if(i%30 == 0):
            plt.imshow(Test_label[i])
            plt.show()
            plt.imshow(Test_input[i])
            plt.show()
            plt.imshow(output_edit)
            plt.show()
            print("PSNR of {}th image".format(PSNR(Test_input[i],Test_label[i])))
        """
    output_list = np.array(output_list) # (91,)
    print(output_list[0].shape)
    print(image_list[0].shape)
    temp_output_list = np.expand_dims(output_list,axis=1)
    temp_image_list = np.transpose(image_list[0][:,:,1:3],(2,0,1))
    print(output_list[0].shape)
    print(temp_image_list.shape)

    output_imagelist = []
    for i in range(len(output_list)):
        cropped_image = np.array(image_list[i][8:-8, 8:-8,:])
        cropped_image = np.transpose(cropped_image[:,:,1:3],(2,0,1))

        temp_output_image = np.array(np.expand_dims(output_list[i],axis=0))
       # plt.imshow(np.array(output_list[i]))
      #  plt.show()
       # plt.imshow(image_list[i][6:-6,6:-6,:])
       # plt.show()

        #set size
     #   print("cropped_image : {}".format(cropped_image.shape))
    #    print("temp_output : {}".format(temp_output_image.shape))
     #   print("sum : {}".format(np.array(cropped_image.shape)+np.array(temp_output_image.shape)))
        size = (np.array(cropped_image.shape)+np.array(temp_output_image.shape))//2

        cropped_image = cropped_image[:,0:size[1],0:size[2]]
        temp_output_image = temp_output_image[:,0:size[1],0:size[2]]

       # print("temp_output_image : {}".format(temp_output_image.shape))
      #  print("cropped_image : {}".format(cropped_image.shape))
        """
        if i % 60 ==0:
            plt.imshow(np.squeeze(temp_output_image,axis = 0))
            plt.show()
            plt.imshow(cropped_image[0,:,:])
            plt.show()
            plt.imshow(cropped_image[1, :, :])
            plt.show()
        """
        concated_image = np.concatenate((temp_output_image,cropped_image),axis=0)
        concated_image = np.transpose(concated_image,(1,2,0))
        plt.imshow(concated_image)
        plt.show()
        #   print(concated_image.shape)
        concated_image = cv2.cvtColor(concated_image,cv2.COLOR_YCR_CB2RGB)
        output_imagelist.append(concated_image)
        plt.imshow(concated_image)
        plt.show()
        break


 #   print(np.array(output_imagelist).shape)
    plt.imshow(output_imagelist[2])
    plt.show()

    avg_PSNR = Tot_PSNR/len(Test_input)
    print("average PSNR : {}".format(avg_PSNR))


 #   print(image_list.shape)
  #  print(Test_dataset.shape)
  #  print(Test_dataset[0])
   # plt.imshow(Test_dataset[0])
  #  plt.show()

   # print(Dataset_test.label)

   # print(Dataset_test)
    #for data in TestDataLoader:
