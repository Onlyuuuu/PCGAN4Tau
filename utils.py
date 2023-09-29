
import numpy  as np
import math


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def mse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse


def mae(img1, img2):
    mae = np.mean(abs(img1 - img2))
    return mae


def ssim(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom


# ## use the scikit package
# from skimage.measure import compare_ssim as ssim
#
# ssim(img1, img2)  # for gray image
# ssim(img1, img1, multichannel=True)  ## for rgb

