from helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import scipy.io as io
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset
from matplotlib import pyplot as plt
from pytorch_ssim import ssim



def get_avg_interp(phantom, pooling_size=[1,1]):
    """
    phantom: (b,1,h,w)
    """
    phantom_size = phantom.shape[-2:]
    avg = nn.AdaptiveAvgPool2d(output_size=pooling_size)(phantom) # (b,1,h*r,w*r)
    if pooling_size[0] == 1 and pooling_size[1] == 1:
        norm_phantom = avg
    else:
        norm_phantom = F.interpolate(avg, size=phantom_size, mode='bilinear', align_corners=True) # (b,1,h,w)
    return norm_phantom


def get_gaussian_blur(phantom, kernel_size=21, sigma=3):
    """
    image: (b,1,h,w)
    二维高斯模糊
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



def get_mask_P(mask_path="./data/prep_data_param/S2Zmatrix.mat", need_P=False):
    """
    return:
        mask: (n_views,n_views)

        P: shape of (c, 280), here c is num of zernike mods, 280 is (num of valid pixels in mask) * 2

        valid_views: list of index of valid views
    """
    data_dict = io.loadmat(mask_path) 

    mask = torch.from_numpy(data_dict["mask"])
    n_views = mask.shape[0]
    valid_views = [i*n_views+j for j in range(n_views) for i in range(n_views) if not mask[i,j].isnan()]

    if need_P:
        P = torch.from_numpy(data_dict["S2Zmatrix"]).type(torch.float32)
    else:
        P = None

    return mask, P, valid_views


def crop_meta_image(meta_image):
    """
    crop meta image to 10x10 times

    e.g. (341,341) -> (340,340)
    """
    meta_shape = (meta_image.shape[-2]//10*10, meta_image.shape[-1]//10*10)
    return meta_image[..., :meta_shape[0], :meta_shape[1]]


def shiftmap_convertor_validviews2fullsize(shiftmap, mask, valid_views):
    """
    shiftmap: (b,m,n,2)

    mask: (n_views,n_views)
    
    valid_views: list of index of valid views

    return: shiftmap_full, (n_views,n_views,m,n,2)
    """
    n_views = mask.shape[0]
    b, m, n, c = shiftmap.shape
    shiftmap_full = torch.zeros(n_views * n_views, m, n, c).to(shiftmap.device)
    shiftmap_full[valid_views,:,:,:] = shiftmap
    shiftmap_full = shiftmap_full.reshape(n_views, n_views, m, n, c)
    return shiftmap_full


def shiftmap_convertor_fullsize2validviews(shiftmap_full, mask, valid_views):
    """
    shiftmap_full: (n_views,n_views,m,n,2)

    mask: (n_views,n_views)
    
    valid_views: list of index of valid views

    return: shiftmap, (b,m,n,2)
    """
    n_views = mask.shape[0]
    _, _, m, n, c = shiftmap_full.shape
    shiftmap = shiftmap_full.reshape(n_views * n_views, m, n, c)[valid_views,:,:,:]
    return shiftmap


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        hidden_size: tuple of hidden layer sizes, might be any length
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(nn.Linear(hidden_size[-1], output_size))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
    

class ShiftMap(nn.Module):
    def __init__(self, batch_size, phantom_size, ctrl_size):
        super(ShiftMap, self).__init__()
        self.batch_size = batch_size
        self.phantom_size = phantom_size # (h,w)
        self.ctrl_size = ctrl_size

        self.trans = (ctrl_size[0] == 1 and ctrl_size[1] == 1)
        self.shiftmap = nn.Parameter(torch.zeros(batch_size, ctrl_size[0], ctrl_size[1], 2), requires_grad=True)  # (b,m,n,2)
        self.mesh = nn.Parameter(torch.stack(torch.meshgrid(torch.linspace(-1,1,self.phantom_size[0]), torch.linspace(-1,1,self.phantom_size[1])),
                                             dim=-1).unsqueeze(0), requires_grad=False) # (1,h,w,2)
        
    def set_shiftmap(self, shiftmap=None):
        """
        shiftmap: (b,m,n,2)
        """
        if shiftmap is None:
            self.shiftmap.data = self.shiftmap.data * 0
        else:
            self.shiftmap.data = self.shiftmap.data * 0 + shiftmap

    def forward(self, input_phantom):
        """
        input_phantom: (b,1,h,w)
        """
        if self.trans:
            shift_mesh = self.mesh + self.shiftmap
        else:
            if self.ctrl_size[0] != self.phantom_size[0] or self.ctrl_size[1] != self.phantom_size[1]:
                shiftmap = F.interpolate(self.shiftmap.permute(0,3,1,2), size=tuple(self.phantom_size), mode='bicubic', align_corners=True).permute(0,2,3,1)
                shift_mesh = self.mesh + shiftmap # (b,h,w,2)
            else:
                shift_mesh = self.mesh + self.shiftmap
        warped_phantom = F.grid_sample(input_phantom, shift_mesh.flip(-1), mode='bicubic', padding_mode='zeros', align_corners=True) # (b,1,h,w)
        return warped_phantom


class PhaseMap(nn.Module):
    def __init__(self, device, phantom, mask_path, ref_view, 
                 norm_type, pooling_size=None, kernel_size=20, sigma=3, ctrl_size=[19,25], loss_crop_ratio=0.98,
                 proj_type="mlp", mlp_path=None, hidden_size=[300,500], num_zernike=35, loss_type="l2"):
        """
        phantom: (n_views^2,1,h,w)
        
        valid_views: list of index of valid views

        ref_view: int, index of reference view

        norm_type: str, type of normalization, can be 'avg' or 'gaussian' or 'none'

        pooling_size: (h,w), size of pooling, if None, set to be (1,1)

        kernel_size: int, size of gaussian kernel

        sigma: float, sigma of gaussian kernel

        ctrl_size: (m,n), size of control tps

        lr1, lr2: float, learning rate for coarse and fine training

        loss_crop_ratio: float, ratio of loss crop, set to 1 for no crop

        proj_type: str, projection type, can be 'mlp' or 'linear'

        hidden_size: list of int, size of hidden layers in MLP

        num_zernike: int, number of zernike modes

        lr_mlp: float, learning rate for MLP

        loss_type: str, type of loss, can be 'l2', 'l1', 'grad', 'ssim'
        """
        super(PhaseMap, self).__init__()
        self.device = device
        self.norm_type = norm_type
        self.pooling_size = pooling_size if pooling_size is not None else [1, 1]
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.ctrl_size = ctrl_size
        self.loss_crop_ratio = loss_crop_ratio
        self.hidden_size = hidden_size
        self.num_zernike = num_zernike
        self.loss_type = loss_type

        # preprocessing for phantoms
        self.set_mask_P_validviews(mask_path, need_P=False)
        self.set_phantom(phantom, ref_view)

        # param for ShiftMap
        self.batch_size = self.input_phantom.shape[0]
        self.phantom_size = self.input_phantom.shape[-2:]

        # shiftmap models
        self.model1 = ShiftMap(self.batch_size, self.phantom_size, ctrl_size=(1,1)).to(self.device)
        self.model2 = ShiftMap(self.batch_size, self.phantom_size, ctrl_size=self.ctrl_size).to(self.device)
        self.crop_transform = transforms.CenterCrop(size=(int(self.phantom_size[0]*loss_crop_ratio), int(self.phantom_size[1]*loss_crop_ratio)))
        if self.loss_type == "l2":
            self.criterion = nn.MSELoss(reduction='mean')
        elif self.loss_type == "l1":
            self.criterion = nn.L1Loss(reduction='mean')
        elif self.loss_type == "grad":
            self.criterion = nn.MSELoss(reduction='mean')
        elif self.loss_type == 'ssim':
            self.criterion = None
        self.loss = []
        self.epoch = []
        
        # projection models
        self.pmodel = MLP(2 * len(self.valid_views), self.hidden_size, self.num_zernike).to(self.device)
        self.set_proj_model(proj_type, mlp_path)
        self.pmodel.eval()

    def set_mask_P_validviews(self, mask_path, need_P=False):
        self.mask_path = mask_path
        self.mask, self.P, self.valid_views = get_mask_P(mask_path, need_P=need_P)

    def set_phantom(self, phantom, ref_view):
        """
        phantom: (n_views^2,1,h,w)
        """
        phantom = crop_meta_image(phantom).to(self.device)
        self.ref_view = ref_view
        self.target_phantom = phantom[self.ref_view:self.ref_view+1, :, :, :].clone()
        self.input_phantom = phantom[self.valid_views, :, :, :].clone()
        if self.norm_type == "gaussian":
            self.input_norm = get_gaussian_blur(self.input_phantom, self.kernel_size, self.sigma)
            self.target_norm = get_gaussian_blur(self.target_phantom, self.kernel_size, self.sigma)
        elif self.norm_type == "avg":
            self.input_norm = get_avg_interp(self.input_phantom, self.pooling_size) # (b,1,h,w)
            self.target_norm = get_avg_interp(self.target_phantom, self.pooling_size) # (b,1,h,w)
        elif self.norm_type == "none":
            self.input_norm = torch.tensor(1.0).to(self.input_phantom.device)
            self.target_norm = torch.tensor(1.0).to(self.input_phantom.device)
        else:
            raise Exception("norm_type must be 'avg' or 'gaussian' or 'none'")
    
    def set_proj_model(self, proj_type, mlp_path):
        self.proj_type = proj_type
        self.mlp_path = mlp_path
        if self.proj_type == "mlp":
            if self.mlp_path is not None and op.exists(self.mlp_path):
                data_dict = torch.load(self.mlp_path)
                self.X_mean, self.X_std = data_dict["X_mean"].to(self.device), data_dict["X_std"].to(self.device)
                self.y_mean, self.y_std = data_dict["y_mean"].to(self.device), data_dict["y_std"].to(self.device)
                self.pmodel.load_state_dict(data_dict["model"])
            else:
                self.X_mean, self.y_mean = 0, 0
                self.X_std, self.y_std = 1, 1

    def reset_shiftmap_models(self):
        self.model1.set_shiftmap()
        self.model2.set_shiftmap()
        self.loss = []
        self.epoch = []

    def train(self, epoch1=10, lr1=1e-2, epoch2=30, lr2=1e-3):
        """
        epoch1, epoch2: int, number of epochs for coarse and fine training

        lr1, lr2: float, learning rate for coarse and fine training
        """
        epoch_id = 0
        self.model1.train()
        optimizer1 = optim.Adam(self.model1.parameters(), lr=lr1)
        for _ in range(epoch1):
            if self.loss_type == "ssim":
                warped_phantom = self.crop_transform(self.model1(self.input_phantom / self.input_norm * self.target_norm))
                target_phantom = self.crop_transform(self.target_phantom)
                loss = 1 - ssim(warped_phantom, target_phantom)
            else:
                warped_phantom = self.crop_transform(self.model1(self.input_phantom / self.input_norm))
                target_phantom = self.crop_transform(self.target_phantom / self.target_norm)
                if self.loss_type in ["l2", "l1"]:
                    loss = self.criterion(warped_phantom, target_phantom)
                elif self.loss_type == "grad":
                    x_kernel = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).to(self.device).type(torch.float32)
                    y_kernel = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).to(self.device).type(torch.float32)
                    warped_grad_x = F.conv2d(warped_phantom, x_kernel, padding=1)
                    warped_grad_y = F.conv2d(warped_phantom, y_kernel, padding=1)
                    target_grad_x = F.conv2d(target_phantom, x_kernel, padding=1)
                    target_grad_y = F.conv2d(target_phantom, y_kernel, padding=1)
                    loss = self.criterion(warped_grad_x, target_grad_x) + self.criterion(warped_grad_y, target_grad_y)
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            self.loss.append(loss.item())
            self.epoch.append(epoch_id)
            epoch_id += 1

        self.model2.set_shiftmap(self.model1.shiftmap.detach().clone())
        optimizer2 = optim.Adam(self.model2.parameters(), lr=lr2)

        self.model1.eval()
        self.model2.train() 
        for _ in range(epoch2):      
            if self.loss_type == "ssim":
                warped_phantom = self.crop_transform(self.model2(self.input_phantom / self.input_norm * self.target_norm))
                target_phantom = self.crop_transform(self.target_phantom)
                loss = 1 - ssim(warped_phantom, target_phantom)
            else:
                warped_phantom = self.crop_transform(self.model2(self.input_phantom / self.input_norm))
                target_phantom = self.crop_transform(self.target_phantom / self.target_norm)
                if self.loss_type in ["l2", "l1"]:
                    loss = self.criterion(warped_phantom, target_phantom)
                elif self.loss_type == "grad":
                    x_kernel = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).to(self.device).type(torch.float32)
                    y_kernel = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).to(self.device).type(torch.float32)
                    warped_grad_x = F.conv2d(warped_phantom, x_kernel, padding=1)
                    warped_grad_y = F.conv2d(warped_phantom, y_kernel, padding=1)
                    target_grad_x = F.conv2d(target_phantom, x_kernel, padding=1)
                    target_grad_y = F.conv2d(target_phantom, y_kernel, padding=1)
                    loss = self.criterion(warped_grad_x, target_grad_x) + self.criterion(warped_grad_y, target_grad_y)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            self.loss.append(loss.item())
            self.epoch.append(epoch_id)
            epoch_id += 1

    def get_loss_curve_shiftmap(self):
        """
        loss curve for shiftmap training
        
        return epoch_list, loss_list
        """
        return self.epoch, self.loss
    
    def get_raw_shiftmap(self, scaling=True):
        """
        scaling: bool, whether to re-scale the shiftmap's x and y to the same unit as y (2/h)

        return raw shiftmap, (b,m,n,2)
        """
        shiftmap = self.model2.shiftmap.detach().clone()
        if scaling and self.phantom_size[0] != self.phantom_size[1]:
            shiftmap[:,:,:,1] = shiftmap[:,:,:,1] * self.phantom_size[1] / self.phantom_size[0]
        return shiftmap

    def get_shiftmap(self, scaling=True):
        """
        return shiftmap with full pupil size and control tps size, (n_views,n_views,m,n,2)

        scaling: bool, whether to re-scale the shiftmap's x and y to the same unit as y (2/h)
        """
        shiftmap = self.get_raw_shiftmap(scaling=scaling)
        shiftmap_full = shiftmap_convertor_validviews2fullsize(shiftmap, self.mask, self.valid_views)
        return shiftmap_full

    def get_warped_phantom(self, test_views):
        """
        here test_views are the views to be tested, not the valid views

        count view_id on pupil size with row first 
        """
        self.model2.eval()
        # raw model output
        with torch.no_grad():
            if self.loss_type == "ssim":
                warped_phantom = self.model2(self.input_phantom / self.input_norm * self.target_norm)
            else:
                warped_phantom = self.model2(self.input_phantom / self.input_norm) * self.target_norm
        # find test views
        valid_test_views = [view_id for view_id in test_views if view_id in self.valid_views]
        id_in_valid_views = [self.valid_views.index(view_id) for view_id in valid_test_views]
        input_phantom = self.input_phantom[id_in_valid_views,:,:,:]
        input_norm =  self.input_norm[id_in_valid_views,:,:,:] if self.norm_type != "none" else self.input_norm
        warped_phantom = warped_phantom[id_in_valid_views,:,:,:]
        target_phantom = self.target_phantom.broadcast_to(input_phantom.shape)
        target_norm = self.target_norm.broadcast_to(input_norm.shape) if self.norm_type != "none" else self.target_norm
        return input_phantom.detach(), warped_phantom.detach(), target_phantom.detach(), input_norm.detach(), target_norm.detach(), valid_test_views

    def get_zernike(self, shiftmap=None, scaling_factor=None):
        """
        return zernike based on shiftmap, (c,m,n)

        scaling_factor: float, phase * scaling_factor = true phase
        """
        if shiftmap is None:
            shiftmap = self.get_raw_shiftmap(scaling=True) # (b,m,n,2)
        else:
            shiftmap = shiftmap.to(self.device)
        b, m, n, _ = shiftmap.shape

        if self.proj_type == "mlp":
            shiftmap = shiftmap.permute(1,2,3,0).reshape(m*n,2*b) # (m*n,2*b)
            shiftmap = (shiftmap - self.X_mean) / self.X_std
            phase = self.pmodel(shiftmap)
            phase = phase * self.y_std + self.y_mean
            phase = phase.reshape(m,n,self.num_zernike).permute(2,0,1)

        elif self.proj_type == "linear":
            # update P
            if self.P is None:
                self.set_mask_P_validviews(self.mask_path, need_P=True)
            # linear projection using self.P
            self.P = self.P.to(self.device)
            shiftmap = shiftmap.permute(1,2,3,0).flip(2).reshape(m,n,2*b)
            phase = F.linear(shiftmap, self.P).permute(2,0,1)
            # scaling
            factor = scaling_factor if scaling_factor is not None else (-1 * self.phantom_size[0] / 380 * 26.5079) 
            phase = phase * factor

        return phase



