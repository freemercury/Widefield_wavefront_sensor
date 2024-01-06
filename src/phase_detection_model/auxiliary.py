import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import scipy.io as io
import math
import numpy as np



def get_avg_interp(phantom, pooling_size=[1,1]):
    """
    phantom: (b,1,h,w)

    return: norm_phantom, (b,1,h,w) or (b,1,1,1) if pooling_size == [1,1]
    """
    assert len(pooling_size) == 2
    assert pooling_size[0] >= 1 and pooling_size[1] >= 1
    assert len(phantom.shape) == 4

    phantom_size = phantom.shape[-2:]
    avg = nn.AdaptiveAvgPool2d(output_size=pooling_size)(phantom) # (b,1,h*r,w*r)
    if pooling_size[0] == 1 and pooling_size[1] == 1:
        norm_phantom = avg
    else:
        norm_phantom = F.interpolate(avg, size=phantom_size, mode='bilinear', align_corners=True) # (b,1,h,w)
    return norm_phantom


def get_gaussian_blur(phantom, kernel_size=21, sigma=3):
    """
    phantom: (b,1,h,w)

    return: norm_phantom (b,1,h,w)
    """
    # gaussian kernel
    x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=np.float32)
    g = np.exp(-x**2 / (2 * sigma**2)) / (math.sqrt(2 * math.pi) * sigma)
    g = g[np.newaxis, :]
    g = g.T @ g
    g /= g.sum()
    g = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(phantom.device)

    # padding
    phantom = F.pad(phantom, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')

    return F.conv2d(phantom, g, padding=0)


def get_mask(mask_path="./data/settings/mask.mat"):
    """
    return:
        mask: (n_view_x,n_view_y)

        valid_views: list of index of valid views
    """
    data_dict = io.loadmat(mask_path) 

    mask = torch.from_numpy(data_dict["mask"])
    n_view_x, n_view_y = mask.shape[0:2]
    valid_views = [i*n_view_y+j for i in range(n_view_x) for j in range(n_view_y) if not mask[i,j].isnan()]

    return mask, valid_views


def shiftmap_convertor_validviews2fullsize(shiftmap, mask, valid_views):
    """
    shiftmap: (n_valid_views,m,n,2)

    mask: (n_view_x,n_view_y)
    
    valid_views: list of index of valid views

    return: shiftmap_full, (n_view_x,n_view_y,m,n,2)
    """
    n_view_x, n_view_y = mask.shape[0:2]
    n_valid_views, m, n, c = shiftmap.shape
    shiftmap_full = torch.zeros(n_view_x * n_view_y, m, n, c).to(shiftmap.device)
    shiftmap_full[valid_views,:,:,:] = shiftmap
    shiftmap_full = shiftmap_full.reshape(n_view_x, n_view_y, m, n, c)
    return shiftmap_full


def shiftmap_convertor_fullsize2validviews(shiftmap_full, mask, valid_views):
    """
    shiftmap_full: (n_view_x,n_view_y,m,n,2)

    mask: (n_views,n_views)
    
    valid_views: list of index of valid views

    return: shiftmap, (n_valid_views,m,n,2)
    """
    n_view_x, n_view_y = mask.shape[0:2]
    _, _, m, n, c = shiftmap_full.shape
    shiftmap = shiftmap_full.reshape(n_view_x * n_view_y, m, n, c)[valid_views,:,:,:]
    return shiftmap
