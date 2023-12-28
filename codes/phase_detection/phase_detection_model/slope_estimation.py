from .auxiliary import *


class SlopeEstimationModule(nn.Module):
    def __init__(self, batch_size, phantom_size, ctrl_size):
        """
        batch_size: int, same as length of valid_views

        phantom_size: (h,w), size of phantom

        ctrl_size: (m,n), number of control pts
        """
        super(SlopeEstimationModule, self).__init__()
        self.batch_size = batch_size
        self.phantom_size = phantom_size # (h,w)
        self.ctrl_size = ctrl_size

        self.trans = (ctrl_size[0] == 1 and ctrl_size[1] == 1)
        self.shiftmap = nn.Parameter(torch.zeros(batch_size, ctrl_size[0], ctrl_size[1], 2), requires_grad=True)
        self.mesh = nn.Parameter(torch.stack(torch.meshgrid(torch.linspace(-1,1,self.phantom_size[0]), torch.linspace(-1,1,self.phantom_size[1])),
                                             dim=-1).unsqueeze(0), requires_grad=False)
        
    def set_shiftmap(self, shiftmap=None):
        """
        shiftmap: (b,m,n,2); if None, set to be zero
        """
        if shiftmap is None:
            self.shiftmap.data = self.shiftmap.data * 0
        else:
            self.shiftmap.data = self.shiftmap.data * 0 + shiftmap

    def forward(self, input_phantom):
        """
        input_phantom: (b,1,h,w)
        """
        if self.trans or (self.ctrl_size[0] == self.phantom_size[0] and self.ctrl_size[1] == self.phantom_size[1]):
            shift_mesh = self.mesh + self.shiftmap
        else:
            shiftmap = F.interpolate(self.shiftmap.permute(0,3,1,2), size=tuple(self.phantom_size), mode='bicubic', align_corners=True).permute(0,2,3,1)
            shift_mesh = self.mesh + shiftmap
        warped_phantom = F.grid_sample(input_phantom, shift_mesh.flip(-1), mode='bicubic', padding_mode='zeros', align_corners=True)
        return warped_phantom



class SlopeEstimation(nn.Module):
    def __init__(self, device, mask_size, phantom_size, phantom, mask_path, ref_view, 
                 norm_type, pooling_size=[1,1], kernel_size=20, sigma=3, ctrl_size=[19,25], loss_type="l2",
                 loss_crop_ratio=0.98, epochs=[20,100], lrs=[1e-2,1e-3]):
        """
        device: torch.device

        mask_size: (n_view_x,n_view_y), size of mask, should NOT be None

        phantom_size: (h,w), size of phantom, should NOT be None

        phantom: (n_views_x*n_view_y,1,h,w), (h,w) should be the same as phantom_size; if None, should use set_phantom to set phantom later

        mask_path: str, path of mask.mat, size of mask should be the same as mask_size; if None, automatically generate a circular mask with the same size as mask_size

        ref_view: int, index of reference view, index between 0 and n_views^2-1

        norm_type: str, type of normalization, can be 'avg' or 'gauss' or None

        pooling_size: (h,w), size of pooling, if None, set to be (1,1)

        kernel_size: int, size of gaussian kernel

        sigma: float, sigma of gaussian kernel

        ctrl_size: (m,n), size of control pts

        loss_type: str, type of loss, can be 'l2', 'l1', 'grad'

        loss_crop_ratio: float in [0, 1], ratio of loss crop, set to 1 for no crop

        epochs: list of int, number of epochs for coarse and fine training

        lrs: list of float, learning rate for coarse and fine training
        """
        super(SlopeEstimation, self).__init__()
        self.device = device
        self.mask_size = mask_size
        self.phantom_size = phantom_size
        self.phantom = phantom
        self.mask_path = mask_path
        self.ref_view = ref_view
        self.norm_type = norm_type
        self.pooling_size = pooling_size if pooling_size is not None else [1, 1]
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.ctrl_size = ctrl_size
        self.loss_type = loss_type
        self.loss_crop_ratio = loss_crop_ratio
        self.epochs = epochs
        self.lrs = lrs

        # set mask and valid views
        self.set_mask_valid_views(self.mask_path)
        self.batch_size = len(self.valid_views)

        # set phantom
        self.set_phantom(self.phantom, self.ref_view)

        # param for ShiftMap
        self.batch_size = self.input_phantom.shape[0]

        # slope estimation models
        self.model1 = SlopeEstimationModule(self.batch_size, self.phantom_size, ctrl_size=(1,1)).to(self.device)
        self.model2 = SlopeEstimationModule(self.batch_size, self.phantom_size, ctrl_size=self.ctrl_size).to(self.device)
        self.crop_transform = transforms.CenterCrop(size=(int(self.phantom_size[0]*loss_crop_ratio), int(self.phantom_size[1]*loss_crop_ratio)))
        if self.loss_type == "l2":
            self.criterion = nn.MSELoss(reduction='mean')
        elif self.loss_type == "l1":
            self.criterion = nn.L1Loss(reduction='mean')
        elif self.loss_type == "grad":
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            raise Exception("loss_type must be 'l2' or 'l1' or 'grad'")
        self.loss = []
        self.epoch = []

    def set_mask_valid_views(self, mask_path):
        """
        mask_path: path to mask.mat, if None, generate a circular mask with the same size as mask_size
        """
        if mask_path is not None:
            self.mask_path = mask_path
            self.mask, self.valid_views = get_mask(self.mask_path)
        else:
            if self.mask_path is not None:
                self.mask, self.valid_views = get_mask(self.mask_path)
            else:
                self.mask = torch.zeros(self.mask_size)
                self.mask[torch.sqrt((torch.arange(self.mask_size[0]) - self.mask_size[0] // 2).pow(2).unsqueeze(1) + (torch.arange(self.mask_size[1]) - self.mask_size[1] // 2).pow(2)) <= self.mask_size[0] // 2] = 1
                self.mask[self.mask < 0.5] = torch.nan
                n_view_x, n_view_y = self.mask_size
                self.valid_views = [i*n_view_y+j for i in range(n_view_x) for j in range(n_view_y) if not self.mask[i,j].isnan()]

    def set_phantom(self, phantom, ref_view):
        """
        phantom: (n_views_x*n_view_y,1,h,w), should NOT be None

        ref_view: int, index between 0 and n_views_x*n_view_y-1
        """
        if phantom is not None:
            self.phantom = phantom.to(self.device)
        else:
            if self.phantom is None:
                self.phantom = torch.randn(self.mask_size[0]*self.mask_size[1], 1, self.phantom_size[0], self.phantom_size[1]).to(self.device) 
        self.ref_view = ref_view
        self.target_phantom = self.phantom[self.ref_view:self.ref_view+1, :, :, :].clone()
        self.input_phantom = self.phantom[self.valid_views, :, :, :].clone()
        if self.norm_type == "gauss":
            self.input_norm = get_gaussian_blur(self.input_phantom, self.kernel_size, self.sigma)
            self.target_norm = get_gaussian_blur(self.target_phantom, self.kernel_size, self.sigma)
        elif self.norm_type == "avg":
            self.input_norm = get_avg_interp(self.input_phantom, self.pooling_size) # (b,1,h,w)
            self.target_norm = get_avg_interp(self.target_phantom, self.pooling_size) # (b,1,h,w)
        elif self.norm_type is None:
            self.input_norm = torch.tensor(1.0).to(self.input_phantom.device)
            self.target_norm = torch.tensor(1.0).to(self.input_phantom.device)
        else:
            raise Exception("norm_type must be 'avg' or 'gauss' or None")

    def reset_shiftmap_models(self):
        self.model1.set_shiftmap()
        self.model2.set_shiftmap()
        self.loss = []
        self.epoch = []

    def train(self, epochs=None, lrs=None):
        """
        epochs: list of int, number of epochs for coarse and fine training, if None, use self.epochs

        lrs: list of float, learning rate for coarse and fine training, if None, use self.lrs
        """
        if epochs is not None:
            self.epochs = epochs
        if lrs is not None:
            self.lrs = lrs
        epoch1, epoch2 = self.epochs
        lr1, lr2 = self.lrs

        def train_batch(model, optimizer):
            warped_phantom = self.crop_transform(model(self.input_phantom / self.input_norm))
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss
            
        epoch_id = 0
        self.model1.train()
        optimizer1 = optim.Adam(self.model1.parameters(), lr=lr1)
        for _ in range(epoch1):
            loss = train_batch(self.model1, optimizer1)
            self.loss.append(loss.item())
            self.epoch.append(epoch_id)
            epoch_id += 1

        self.model2.set_shiftmap(self.model1.shiftmap.detach().clone())
        optimizer2 = optim.Adam(self.model2.parameters(), lr=lr2)
        self.model1.eval()
        self.model2.train() 
        for _ in range(epoch2): 
            loss = train_batch(self.model2, optimizer2)
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
    
    def shiftmap_v2f(self, shiftmap):
        """
        shiftmap: (n_valid_views,m,n,2)

        return: shiftmap_full, (n_view_x,n_view_y,m,n,2)
        """
        return shiftmap_convertor_validviews2fullsize(shiftmap, self.mask, self.valid_views)
    
    def shiftmap_f2v(self, shiftmap_full):
        """
        shiftmap_full: (n_view_x,n_view_y,m,n,2)

        return: shiftmap, (n_valid_views,m,n,2)
        """
        return shiftmap_convertor_fullsize2validviews(shiftmap_full, self.mask, self.valid_views)

    def get_warped_phantom(self, test_views):
        """
        here test_views are the views to be tested, not the valid views

        count view_id on pupil size with row first 
        """
        self.model2.eval()
        # raw model output
        with torch.no_grad():
            warped_phantom = self.model2(self.input_phantom / self.input_norm) * self.target_norm
        # find test views
        valid_test_views = [view_id for view_id in test_views if view_id in self.valid_views]
        id_in_valid_views = [self.valid_views.index(view_id) for view_id in valid_test_views]
        input_phantom = self.input_phantom[id_in_valid_views,:,:,:]
        input_norm =  self.input_norm[id_in_valid_views,:,:,:] if self.norm_type is not None else self.input_norm
        warped_phantom = warped_phantom[id_in_valid_views,:,:,:]
        target_phantom = self.target_phantom.broadcast_to(input_phantom.shape)
        target_norm = self.target_norm.broadcast_to(input_norm.shape) if self.norm_type is not None else self.target_norm
        return input_phantom.detach(), warped_phantom.detach(), target_phantom.detach(), input_norm.detach(), target_norm.detach(), valid_test_views

