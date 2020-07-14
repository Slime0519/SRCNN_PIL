import torch.utils.data as data
import torch
import h5py
import numpy as np

class Read_dataset_h5(data.Dataset):
    def __init__(self,filepath):
        super(Read_dataset_h5, self).__init__()
        h5_file = h5py.File(filepath)
        self.input = np.array(h5_file.get('lr'))
        self.label = np.array(h5_file.get('hr'))

    def __getitem__(self, index):
        self.torchinput = torch.from_numpy(np.expand_dims(self.input[index],0).astype('float32'))
        self.torchinput /=255.
        self.torchlabel = torch.from_numpy(np.expand_dims(self.label[index],0).astype('float32'))
        self.torchlabel /= 255.
      #  print(self.input.shape[0])
      #  print((self.torchinput.numpy()).shape)
        return self.torchinput, self.torchlabel

    def __len__(self):
        return self.input.shape[0]

class Read_dataset_h5_Test(data.Dataset):
    def __init__(self,filepath):
        super(Read_dataset_h5_Test, self).__init__()
        h5_file = h5py.File(filepath)
        self.input = np.array([(h5_file.get('lr').get('{}'.format(i))) for i in range(5)])
        self.label = np.array([(h5_file.get('hr').get('{}'.format(i))) for i in range(5)])
        #print("inputshape : {}".format(np.array(self.input[0]).astype('float32')/255.))
        #print("labelshape : {}".format(self.label.shape))

    def __getitem__(self, index):
        self.torchinput = torch.from_numpy(np.expand_dims(self.input[index],0).astype('float32'))
        self.torchinput /=255.
        self.torchlabel = torch.from_numpy(np.expand_dims(self.label[index],0).astype('float32'))
        self.torchlabel /= 255.
      #  print(self.input.shape[0])
      #  print((self.torchinput.numpy()).shape)
        return self.torchinput, self.torchlabel

    def __len__(self):
        return self.input.shape[0]