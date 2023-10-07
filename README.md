#PCGAN4Tau

These techniques (PCGAN4Tau) are described in the paper:

Jie Sun, Le Xue, Jiaying Lu, Qi Zhang, Shuoyan Zhang, Luyao Wang, Min Wang, for the Alzheimer’s Disease Neuroimaging Initiative, Chuantao Zuo, Jiehui Jiang, Mei Tian

Perception-Enhanced Generative Adversarial Network for Synthesizing Tau Positron Emission Tomography images from Structural Magnetic Resonance Images: a cross-center and cross-tracer study


# Demo

The following commands train and test PCGAN4Tau models for sMRI to tau-PET synthesis on images from the ADNI dataset or private datasets. Copy the "datasets" folder in your current directory.
To run the code on other datasets, create a file named 'data.mat' for training, testing and validation samples and place them in their corresponding directories (datasets/yourdata/train, test, val). 'data.mat' should contain a variable named data_x for the source contrast and data_y for the target contrast. If you are creating the 'data.mat' file via Matlab please make sure that dimensions (1, 2, 3, 4) correspond to (neighbouring slices, number of samples, x-size, y-size). If you are saving the file via python then transpose the dimensions. Also, make sure that voxel intensity of each subject is normalized between 0-1.

## PCGAN4Tau

### Training
python PCGAN4Tau.py --dataroot datasets/ADNI --name pGAN_run --which_direction AtoB --lambda_A 2  --batchSize 8 --output_nc 1 --input_nc 1 --gpu_ids 0 --niter 50 --niter_decay 50 --save_epoch_freq 25 --lambda_vgg 1 --checkpoints_dir checkpoints/ --training

name - name of the experiment 
which_direction - direction of synthesis. If it is set to 'AtoB' synthesis would be from data_x to data_y, and vice versa 
lambda_A - weighting of the pixel-wise loss function 
input_nc, output_nc - number of neighbouring slices used. If you do not want to use the neighboring slices just set them to 1, the central slices would be selected.  
niter, n_iter_decay - number of epochs with normal learning rate and number of epochs for which the learning leate is decayed to 0. Total number of epochs is equal to sum of them 
save_epoch_freq -frequency of saving models
lambda_vgg - weighting of the dual perceptual loss function 

### Testing
python PCGAN4Tau.py --dataroot datasets/ADNI --name pGAN_run --which_direction AtoB --phase test --output_nc 1 --input_nc 1 --how_many 1 --results_dir results/ --checkpoints_dir checkpoints/


## Prerequisites
Windows
Python 3.8
NVIDIA GPU + CUDA CuDNN
Pytorch [1.7.1]
Other dependencies - visdom, dominate  

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.



## Acknowledgments
This code is based on implementations by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan), [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [pGAN-cGAN](https://github.com/icon-lab/pGAN-cGAN).
