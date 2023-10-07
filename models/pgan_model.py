import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np

import matplotlib.pyplot as plt
from torchvision import models

class pGAN(BaseModel):
    def name(self):
        return 'pGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(1, opt.output_nc, opt.ngf,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids).cuda()
      
        self.vgg16=VGG16().cuda()
        self.vgg19=VGG19().cuda()
        self.KLDLoss=KLDLoss().cuda()
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids).cuda()
        if not self.isTrain or opt.continue_train:    ##
            self.load_network(self.netG, 'G', opt.which_epoch)   ##
            
            if self.isTrain:    ##
                self.load_network(self.netD, 'D', opt.which_epoch)   ###

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionKL = torch.nn.KLDivLoss()
            self.lambda_kl = 0.4  # Weight for KL divergence loss
            
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        BtoA = self.opt.which_direction == 'BtoA'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        # if len(self.gpu_ids) > 0:
        #     input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
        #     input_B = input_B.cuda(self.gpu_ids[0],non_blocking=True)
        if len(self.gpu_ids) > 0:  ##
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)  ##between
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A).cuda().float()
        self.fake_B = self.netG(self.real_A).cuda().float()##.cuda().float()
        self.real_B = Variable(self.input_B).cuda().float()

    
    def test(self):
        # no backprop gradients
        
        self.real_A = Variable(self.input_A, volatile=True).cuda().float()
        self.fake_B = self.netG(self.real_A.float()).cuda().float()
        self.real_B = Variable(self.input_B, volatile=True).cuda().float()
##added


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B  
        # 条件GAN 将真实输入self.real_A和生成的输入self.fake_B在通道维度上连接起来，形成fake_AB。
        # 然后，将fake_AB传入判别器self.netD中得到对生成样本的判别结果pred_fake。
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A.cuda().float(), self.fake_B.cuda().float()), 1).data)
        pred_fake = self.netD(fake_AB.cuda())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # RealAB
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D.backward()

        
    def backward_G(self):
        # First, G(A) should fake the discriminator

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        #Perceptual loss  VGG16
        self.VGG16_real=self.vgg16(self.real_B.cuda().float().expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG16_fake=self.vgg16(self.fake_B.cuda().float().expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG16_loss=self.criterionL1(self.VGG16_fake,self.VGG16_real)* self.opt.lambda_vgg
        #Perceptual loss  VGG19
        self.VGG19_real=self.vgg19(self.real_B.cuda().float().expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG19_fake=self.vgg19(self.fake_B.cuda().float().expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG19_loss=self.criterionL1(self.VGG19_fake,self.VGG19_real)* self.opt.lambda_vgg
                
        
        self.loss_vgg = self.VGG16_loss + self.VGG19_loss
        #target_distribution = self.real_B  # 使用真实输入作为目标分布
        #fake_distribution = self.fake_B  # 使用生成器输出作为生成分布
        #self.loss_kl = self.criterionKL(fake_distribution,target_distribution)

    # Overall Objective Loss
        #epsilon = 1e-8  # 添加一个小的偏置项，避免除以零
        #mu = self.fake_B.mean()  # 计算生成器输出的均值
        #logvar = (self.fake_B.std().pow(2) + epsilon).log()  # 计算生成器输出的对数方差
        #self.loss_kl = self.KLDLoss(mu, logvar)
        self.idt_A = self.netG(self.real_B)  ##输入 B 并将B转为自身
        self.loss_idt = self.criterionL1(self.idt_A, self.real_B) * self.opt.lambda_identity ##计算这个差值
            # G_B should be identity if real_A is fed: ||G_B(A) - A||

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_vgg  +self.loss_idt
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('Total_G', self.loss_G.item()),  
                            ('Total_D', self.loss_D.item()),##item()
                            ('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('G_VGG16', self.VGG16_loss.item()),
                            ('G_VGG19', self.VGG19_loss.item()),
                            ('G_VGG', self.loss_vgg.item()),
                            ('G_idt', self.loss_idt.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
    
    
    
    
    
    

      

#Extracting VGG feature maps before the 2nd maxpooling layer  
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, X):
        h_relu1 = self.stage1(X)
        h_relu2 = self.stage2(h_relu1)       
        return h_relu2
    
import torchvision.models as models

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        for x in range(9):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, X):
        h_relu1 = self.stage1(X)
        h_relu2 = self.stage2(h_relu1)
        return h_relu2

class KLDLoss(torch.nn.Module):
    def __init__(self, ):
        super(KLDLoss, self).__init__()
    
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
