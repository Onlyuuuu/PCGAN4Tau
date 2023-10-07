 
import io
import scipy.io as scio
import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np, h5py
import random
import hdf5storage

##创建数据集加载主函数
def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())    ### 返回的名字为“CustomDatasetDataLoader”
    data_loader.initialize(opt)   ###初始化参数
    return data_loader

 ##创建数据集
def CreateDataset(opt):
    # data directory
    target_file = opt.dataroot + '/' + opt.phase + '/data.mat'
    f = hdf5storage.loadmat(target_file)
    ##f = h5py.File(target_file, 'r')
  
    # f = scio.loadmat(target_file, 'r')
    #f = scio.loadmat(target_file)
    f['data_x'] = np.transpose(f['data_x'], (2, 3, 1, 0))
    f['data_y'] = np.transpose(f['data_y'], (2, 3, 1, 0))
    print(np.array(f['data_x']).shape,np.array(f['data_y']).shape)
    slices = np.array(f['data_x']).shape[3] / 2
    samples = list(range(np.array(f['data_y']).shape[2]))
    # Selecting neighbouring slices based on the inputs
    if opt.which_direction == 'AtoB':
        data_x = np.array(f['data_x'])[:, :, :, int(slices - opt.input_nc / 2):int(slices + opt.input_nc / 2 + 1)]  ##int
        data_y = np.array(f['data_y'])[:, :, :,int(slices - opt.output_nc / 2):int(slices + opt.output_nc / 2 + 1)]
    else:
        data_y = np.array(f['data_y'])[:, :, :, int(slices - opt.input_nc / 2):int(slices + opt.input_nc / 2 + 1)]
        data_x = np.array(f['data_x'])[:, :, :, int(slices - opt.output_nc / 2):int(slices + opt.output_nc / 2 + 1)]
    # Shuffle slices in data_y for the cGAN case (incase the input data is registered)
    if opt.dataset_mode == 'unaligned_mat':
        if opt.isTrain:
            print("Training phase")
            random.shuffle(samples)
        else:
            print("Testing phase")
        data_y = data_y[:, :, samples, :]
    data_x = np.transpose(data_x, (3, 2, 0, 1))
    data_y = np.transpose(data_y, (3, 2, 0, 1))
    # Ensure that there is no value less than 0
    data_x[data_x < 0] = 0
    data_y[data_y < 0] = 0
    dataset = []
    # making range of each image -1 to 1 and converting to torch tensor
    for train_sample in range(data_x.shape[1]):
        data_x[:, train_sample, :, :] = (data_x[:, train_sample, :, :] - 0.5) / 0.5
        data_y[:, train_sample, :, :] = (data_y[:, train_sample, :, :] - 0.5) / 0.5
        print(data_x.shape,data_y.shape)
        dataset.append(
            {'A': torch.from_numpy(data_x[:, train_sample, :, :]), 'B': torch.from_numpy(data_y[:, train_sample, :, :]),
             'A_paths': opt.dataroot, 'B_paths': opt.dataroot})
    # If number of samples in data_x and data_y are different (for unregistered images) make them same
    if data_x.shape[1] != data_y.shape[1]:
        for train_sample in range(max(data_x.shape[1], data_y.shape[1])):
            if data_x.shape[1] >= data_y.shape[1] and train_sample > (data_y.shape[1] - 1):
                data_x[:, train_sample, :, :] = (data_x[:, train_sample, :, :] - 0.5) / 0.5
                dataset.append({'A': torch.from_numpy(data_x[:, train_sample, :, :]), 'A_paths': opt.dataroot})
            elif data_y.shape[1] > data_x.shape[1] and train_sample > (data_x.shape[1] - 1):
                data_y[:, train_sample, :, :] = (data_y[:, train_sample, :, :] - 0.5) / 0.5
                dataset.append({'B': torch.from_numpy(data_y[:, train_sample, :, :]), 'B_paths': opt.dataroot})
            else:
                data_x[:, train_sample, :, :] = (data_x[:, train_sample, :, :] - 0.5) / 0.5
                data_y[:, train_sample, :, :] = (data_y[:, train_sample, :, :] - 0.5) / 0.5
                dataset.append({'A': torch.from_numpy(data_x[:, train_sample, :, :]),
                                'B': torch.from_numpy(data_y[:, train_sample, :, :]),
                                'A_paths': opt.dataroot, 'B_paths': opt.dataroot})

    return dataset

##加载数据集
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)  ##初始化参数
        self.dataset = CreateDataset(opt)     ##创建数据集
        self.dataloader = torch.utils.data.DataLoader(  ##加载创建好的数据集，并定义相关参数
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
             )

    def load_data(self):
        return self.dataloader      ##返回数据集

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)  ##返回加载的数据集长度和一个epoch容许的加载最大容量

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
