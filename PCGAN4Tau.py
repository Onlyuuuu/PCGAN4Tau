from multiprocessing import freeze_support
import torch
import numpy as np
from utils import psnr, mae, ssim

torch.cuda.set_device(-1)
##train运行时会进行前向传播和反向传播  test模型仅仅前向传播
def train():
    import time
    from options.train_options import TrainOptions
    from data import CreateDataLoader
    from models import create_model
    from util.visualizer import Visualizer
    opt = TrainOptions().parse()
    model = create_model(opt)
    #加载数据集
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('Training images = %d' % dataset_size)    
    visualizer = Visualizer(opt)
    total_steps = 0
    #Starts training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            #Save current images (real_A, real_B, fake_B)
            if  epoch_iter % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch,epoch_iter, save_result)
            #Save current errors   
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)
            #Save model based on the number of iterations
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
    
            iter_data_time = time.time()
        #Save model based on the number of epochs
        print(opt.dataset_mode)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
    
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        test()
    
tranfile = open('1.txt','w',encoding='utf-8')
def test():
    from skimage.metrics import mean_squared_error as compare_mse
    #from sklearn.metrics import mean_absolute_error as compare_mae
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
    import sys
    sys.argv=args  
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    from options.test_options import TestOptions
    from data import CreateDataLoader
    from models import create_model
    from util.visualizer import Visualizer
    from util import html
    import scipy.io as io
    
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
      # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        fake_B = (model.fake_B.cpu().detach().numpy() * 0.5 + 0.5) * 255  
        real_B = (model.real_B.cpu().detach().numpy() * 0.5 + 0.5) * 255  
        real_A = (model.real_A.cpu().detach().numpy() * 0.5 + 0.5) * 255
        real_B = np.transpose(real_B, (2, 3, 1, 0))[:, :, :, 0] ##
        fake_B = np.transpose(fake_B, (2, 3, 1, 0))[:, :, :, 0]
        real_A = np.transpose(real_A, (2, 3, 1, 0))[:, :, :, 0]
        # cv2.cv2.imwrite('./1.png',fake_B)
        psnr_sum = 0.0
        mae_sum = 0.0
        mse_sum = 0.0
        ssim_sum = 0.0
        num_files = 0
        #PSNR = psnr(real_A, fake_B)
        MAE = mae(real_B, fake_B)
        #SSIM = ssim(real_B, fake_B)
        # print('psnrscore: ', psnrscore, 'maescore: ', maescore, 'ssimscore: ', ssimscore)
        PSNR = compare_psnr(real_B.astype(np.uint8), fake_B.astype(np.uint8))
        MSE = compare_mse(real_B.astype(np.uint8), fake_B.astype(np.uint8))
        SSIM = compare_ssim(real_B.astype(np.uint8), fake_B.astype(np.uint8), multichannel=True)
        tranfile.write(' PSNR: '+str( PSNR)+'               MAE: '+str( MAE) + '               MSE: '+str( MSE)+ '               SSIM: '+str( SSIM)+'\n')
        print('  PSNR:  ', PSNR, '  MAE:  ', MAE ,'  MSE:  ', MSE, '  SSIM:  ', SSIM)
        psnr_sum += PSNR
        mae_sum += MAE
        mse_sum += MSE
        ssim_sum += SSIM
        num_files +=1
        
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        img_path[0]=img_path[0]+str(i)
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)    
    webpage.save()  
    
    psnr_avg = psnr_sum / num_files
    mae_avg = mae_sum / num_files
    mse_avg = mse_sum / num_files
    ssim_avg = ssim_sum   / num_files
    # 在终端输出平均值
    print("Average PSNR:", psnr_avg)
    print("Average MAE:", mae_avg)
    print("Average MSE:", mse_avg)
    print("Average SSIM:", ssim_avg)
  

if __name__=='__main__':
    import sys
    sys.argv.extend(['--model','pGAN'])
    args=sys.argv
    freeze_support()
    if '--training' in str(args):
        train()
    else:
        sys.argv.extend(['--serial_batches'])
        test()    