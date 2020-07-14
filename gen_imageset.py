import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
import glob
import six
import tarfile

#CROP_SIZE = 32
STRIDE = 14

def load_img(filepath):
    image = Image.open(filepath).convert('YCbCr')
    y,_,_ = image.split()

    return y


def download_bsd300(dest="dataset"):
    output_image_dir = os.path.join(dest, "BSDS300/images")

    if not os.path.exists(output_image_dir):
        os.makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = six.moves.urllib.request.urlopen(url)

        file_path = os.path.join(dest, os.path.basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        os.remove(file_path)

    return output_image_dir

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform(imageset,scaling_factor =2 ):
    if scaling_factor ==3:
        CROP_SIZE =33
    else:
        CROP_SIZE=32
    print(imageset.shape)
    input_patches = np.zeros((0,32,32))
    target_patches = np.zeros((0,32,32))
    for image in imageset:
        image_width = image.shape[0]
        image_height = image.shape[1]
        for i in range(0,image_height-CROP_SIZE,STRIDE):
            for j in range(0,image_width-CROP_SIZE,STRIDE):
                cropped_patch = np.copy(image[i:i+CROP_SIZE,j:j+CROP_SIZE])
                cropped_patch_expanded = np.expand_dims(cropped_patch,axis=0)
                print(cropped_patch.shape)
                print(target_patches.shape)
                target_patches = np.append(target_patches,cropped_patch_expanded,axis=0)

                #downscaling and upscaling
                downscaled_patch = cv2.resize(cropped_patch,dsize=(int(CROP_SIZE/scaling_factor),int(CROP_SIZE/scaling_factor)),interpolation=cv2.INTER_CUBIC)
                blurred_patch = cv2.GaussianBlur(downscaled_patch,(5,5),0)
                recoverd_patch = np.copy(cv2.resize(blurred_patch,dsize=(CROP_SIZE,CROP_SIZE),interpolation=cv2.INTER_CUBIC))
                recoverd_patch_expanded = np.expand_dims(recoverd_patch,axis=0)
                input_patches = np.append(input_patches,recoverd_patch_expanded)

    return input_patches, target_patches


class DatasetGenerator(Dataset):
    def __init__(self,dirpath,  scaling_factor, train = 1):
        super(DatasetGenerator, self).__init__()
        if train == 1:
            dirpath = os.path.join(dirpath, "train")
        else:
            dirpath = os.path.join(dirpath,"test")
       # print(dirpath)
        self.image_filenames = glob.glob(os.path.join(dirpath,"*.jpg"))
        self.image_filenames += (glob.glob(os.path.join(dirpath,"*.bmp")))

        crop_size = calculate_valid_crop_size(32,upscale_factor=scaling_factor)
        """
        self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                                 #  transforms.Resize(crop_size//scaling_factor),
                                                  # transforms.Resize(crop_size,interpolation=Image.BICUBIC),
                                                   transforms.Scale(crop_size),
                                                   transforms.ToTensor()])
        """
        self.target_transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                                    transforms.ToTensor()])

        self.inputimagelist = []
       # print(self.image_filenames)
        for filename in self.image_filenames:
            tempimage = np.array(load_img(filename))
           # print(filename)
            self.inputimagelist.append(tempimage)

        #self.inputimagelist = load_img(self.image_filenames)
        self.inputimagelist = np.array(self.inputimagelist)
       # self.inputimagelist = self.inputimagelist.astype(float)
        self.inputimagelist /=255.
        self.inputlist, self.targetlist = input_transform(self.inputimagelist)
        #print(self.inputimagelist)

    def __getitem__(self, index):
        #input = load_img(self.image_filenames[index])
        #inputlist, targetlist = input_transform(self.inputimagelist)
       # print(np.array(input).shape)

        #input = input.filter(ImageFilter.GaussianBlur(2))
        #input = input_transform(input)
        #target = self.target_transform(target)
        inputdata = np.copy(self.inputlist[index])
        targetdata = np.copy(self.targetlist[index])
        return inputdata, targetdata

    def __len__(self):
        return len(self.image_filenames)


if __name__ == "__main__":
    dirpath = download_bsd300()
    DatasetGenerator(dirpath,2)