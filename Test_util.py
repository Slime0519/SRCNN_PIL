import numpy as np
import h5py_dataset
import h5py
import matplotlib.pyplot as plt

if __name__ =="__main__":
   # dataset  = h5py_dataset.Read_dataset_h5("Dataset/91-image_x2.h5")
    h5_file = h5py.File("Dataset/91-image_x2.h5")
    print(list(h5_file.keys()))
    input = np.array(h5_file.get('lr'))
    label = np.array(h5_file.get('hr'))

    print(input.shape)
    print(label.shape)

    print(input[0]/255.)
    plt.imshow(input[0])
    plt.show()
    plt.imshow(label[0])
    plt.show()


    test_dataset = h5py_dataset.Read_dataset_h5_Test("Dataset/Set5_x2.h5")
    h5_file = h5py.File("Dataset/Set5_x2.h5",'r+')
    print(list(h5_file.keys()))

    input = np.array(h5_file.get('lr').get)
    label = np.array(h5_file.get('hr').get('1'))


    print(input)
    print(label.shape)

    print(input / 255.)
    plt.imshow(input)
    plt.show()
    plt.imshow(label)
    plt.show()
    #plt.imshow(input[0])
    #plt.show()
   # plt.imshow(label[0])
    #plt.show()



