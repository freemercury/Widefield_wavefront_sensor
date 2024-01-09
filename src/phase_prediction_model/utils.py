from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, re, glob




class ZernikeDataset(Dataset):
    """
    Dataset for zernike phase prediction
    
    Class Variables:
    -----------
    data_list: [(n1,c,h,w),(n2,c,h,w),...]
        all zernike tensors
    file_list: [file_list1, file_list2, ...]
        all file_list, match data_list
    train_id_list: [(id1,id2)]
        id1: set_id, id2: file_id
    test_id_list: [(id1,id2)]
        id1: set_id, id2: file_id
    norm: scalar
        max(abs(data_list))
    Z2P, Z2p, p2Z: tensors, 
        shape of (zernike_num,phase_size,phase_size), (zernike_num,valid_phase_size), (valid_phase_size,zernike_num)

    APIs:
    -----------
    reset: reset all class variables
    load_zernike_phase: load Z2P, Z2p, p2Z
    load_data: load data_list, file_list, train_id_list, test_id_list, norm
    zernike2phase: zernike (b,t,c,h,w) -> phase (b,t,p,h,w)
    phase2zernike: phase (b,t,p,h,w) -> zernike (b,t,c,h,w)  
    """
    data_list = []   # [(n1,c,h,w),(n2,c,h,w),...], all zernike
    file_list = []  # [file_list1, file_list2, ...], all files
    train_id_list = []    # [(id1,id2)]
    test_id_list = []    # [(id1,id2)]
    norm = None   # scalar
    Z2P, Z2p, p2Z = None, None, None

    def __init__(self, device, train=True, **kwargs):
        """
        Initialize ZernikeDataset

        Parameters:
        -----------
        device: torch.device
        train: bool
            True for train, False for test
        kwargs: dict
            kwargs["dataset"]:
                data_path: str, path to data
                zernike_phase_path: str, path to zernike_phase%d.mat
                phase_size: int, phase_size
                n_channel: list, [min_channel, max_channel], choose zernike modes, 0 for piston
                split: float, split ratio for train/whole
                t_series: int, length of input sequence
                t_offset: int, offset between input and target
                t_down: int, downsample interval, 0 for no downsample, 1 for 1/2 downsample (interval is 1 frame)
                all_patch: list, [min_h, max_h, min_w, max_w], choose patch for all data, used as indices [min_h:max_h, min_w:max_w]
                input_patch: list, [min_h, max_h, min_w, max_w], choose patch for input data, used as indices [min_h:max_h, min_w:max_w]
                target_patch: list, [min_h, max_h, min_w, max_w], choose patch for target data, used as indices [min_h:max_h, min_w:max_w]
                test_size: list, [h,w], test_size for metric calculation
        """
        super().__init__()

        self.device = device
        self.train = train
        self.kwargs = kwargs

        self.data_path = self.kwargs["dataset"]["data_path"]
        self.zernike_phase_path = self.kwargs["dataset"]["zernike_phase_path"]
        self.phase_size = self.kwargs["dataset"]["phase_size"]
        self.n_channel = list(range(self.kwargs["dataset"]["n_channel"][0], self.kwargs["dataset"]["n_channel"][1]+1))
        self.split = self.kwargs["dataset"]["split"]

        self.t_series = self.kwargs["dataset"]["t_series"]
        self.t_offset = self.kwargs["dataset"]["t_offset"]
        self.t_down = self.kwargs["dataset"]["t_down"]

        self.all_patch = self.kwargs["dataset"]["all_patch"]
        self.input_patch = self.kwargs["dataset"]["input_patch"]
        self.target_patch = self.kwargs["dataset"]["target_patch"]
        self.test_size = self.kwargs["dataset"]["test_size"]

        self.seq_len = (self.t_series + self.t_offset) + (self.t_series + self.t_offset - 1) * self.t_down

        if ZernikeDataset.norm is None:
            ZernikeDataset.load_zernike_phase(self.device, self.zernike_phase_path, self.phase_size, self.n_channel)
            ZernikeDataset.load_data(self.device, self.data_path, self.n_channel, self.seq_len, self.split)
    
    def __len__(self):
        return len(ZernikeDataset.train_id_list) if self.train else len(ZernikeDataset.test_id_list)
    
    def __getitem__(self, idx):
        if self.train:
            id1, id2 = ZernikeDataset.train_id_list[idx]
        else:
            id1, id2 = ZernikeDataset.test_id_list[idx]
        
        input_id_index = list(range(id2, id2 + self.seq_len - self.t_offset * (self.t_down + 1), self.t_down + 1))
        input = ZernikeDataset.data_list[id1][input_id_index,:,self.input_patch[0]:self.input_patch[1],self.input_patch[2]:self.input_patch[3]].clone()  # (t,c,h,w)
        if self.t_offset == 0:
            crop_patch = [self.target_patch[0] - self.input_patch[0], self.target_patch[1] - self.input_patch[0], 
                            self.target_patch[2] - self.input_patch[2], self.target_patch[3] - self.input_patch[2]]
            input[:, :, crop_patch[0]:crop_patch[1], crop_patch[2]:crop_patch[3]] = 0.0

        target_id_index = list(range(id2 + self.t_offset * (self.t_down + 1), id2 + self.seq_len, self.t_down + 1))
        target = ZernikeDataset.data_list[id1][target_id_index,:,self.target_patch[0]:self.target_patch[1],self.target_patch[2]:self.target_patch[3]].clone()
        target_file = ZernikeDataset.file_list[id1][target_id_index[-1]]
        
        return input, target, target_file

    @classmethod
    def reset(cls):
        """
        Reset all class variables
        """
        cls.data_list = []
        cls.file_list = []
        cls.train_id_list = []
        cls.test_id_list = []
        cls.norm = None
        cls.Z2P, cls.Z2p, cls.p2Z = None, None, None

    @classmethod
    def load_zernike_phase(cls, device, zernike_phase_path, phase_size, n_channel):
        """
        load Z2P, Z2p, p2Z

        Parameters:
            device: torch.device
            zernike_phase_path: str, path to zernike_phase%d.mat
            phase_size: int, phase_size
            n_channel: list, choose zernike modes, 0 for piston
        """
        trans_dict = loadmat(zernike_phase_path + "/zernike_phase%d.mat" % (phase_size))
        cls.Z2P = torch.from_numpy(trans_dict["Z2P"]).type(torch.float32).to(device)[n_channel,:,:]
        cls.Z2p = torch.from_numpy(trans_dict["Z2p"]).type(torch.float32).to(device)[n_channel,:]
        cls.p2Z = torch.from_numpy(trans_dict["p2Z"]).type(torch.float32).to(device)[:,n_channel]
    
    @classmethod
    def load_data(cls, device, data_path, n_channel, seq_len, split):
        """
        load data_list, file_list, train_id_list, test_id_list; calculate norm

        Parameters:
            device: torch.device
            data_path: str, path to data
            n_channel: list, choose zernike modes, 0 for piston
            seq_len: int, frames that input and target cover in total
            split: float, split ratio for train/whole
        """
        def extract_number(filename, pattern):
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
            return 0

        # find sets
        set_list = os.listdir(data_path)
        set_list.sort(key=lambda x: extract_number(x, r'.*?set(\d+).*?'))
        
        # update data_list, file_list, train_id_list, test_id_list
        for set_id, set_name in enumerate(set_list):
            set_path = data_path + "/" + set_name + "/"
            file_list = glob.glob(set_path + "/**/*_ds_zernike.mat", recursive=True)
            if len(file_list) != 0:
                file_list.sort(key=lambda x: extract_number(x.split("\\")[-1], r'.*?(\d+).*?'))
                tmp_data = torch.stack([torch.from_numpy(loadmat(file)["zernike"]).type(torch.float32).to(device)[n_channel,:,:] for file in file_list], dim=0)
                cls.data_list.append(tmp_data)
                cls.file_list.append(file_list)
                train_len = int(len(file_list) * split)
                cls.train_id_list += [(set_id, i) for i in range(train_len - seq_len + 1)]
                cls.test_id_list += [(set_id, i) for i in range(train_len - seq_len + 1, len(file_list) - seq_len + 1)]
            else:
                cls.data_list.append(None)
                cls.file_list.append(None)
        
        # update norm
        cls.norm = torch.amax(torch.abs(torch.cat([x for x in cls.data_list if x is not None], dim=0))).item()

    @classmethod
    def zernike2phase(cls, zernike_data):
        """
        Convert zernike to phase

        Parameters:
            zernike_data: (b,t,c,h,w)
        
        Return:
            phase_data: (b,t,p,h,w)
        """
        phase_data = F.linear(zernike_data.permute(0,1,3,4,2), cls.Z2p.permute(1,0), bias=None).permute(0,1,4,2,3)
        return phase_data
    
    @classmethod
    def phase2zernike(cls, phase_data):
        """
        Convert phase to zernike

        Parameters:
            phase_data: (b,t,p,h,w)

        Return:
            zernike_data: (b,t,c,h,w)
        """
        zernike_data = F.linear(phase_data.permute(0,1,3,4,2), cls.p2Z.permute(1,0), bias=None).permute(0,1,4,2,3)
        return zernike_data



class LossFn(nn.Module):
    """
    Loss Function for phase prediction
    """
    def __init__(self, **kwargs):
        """
        Initialize LossFn

        Parameters:
        -----------
        kwargs: dict
            kwargs["criterion"]:
                loss_type: str, 'l2', 'l1'
                reduction: str, 'mean', 'sum'
        """
        super(LossFn, self).__init__()
        self.kwargs = kwargs

        self.loss_type = self.kwargs["criterion"]["loss_type"]
        self.reduction = self.kwargs["criterion"]["reduction"]

        if self.loss_type == "l2":
            self.criterion = nn.MSELoss(reduction=self.reduction)
        elif self.loss_type == "l1":
            self.criterion = nn.L1Loss(reduction=self.reduction)
    
    def forward(self, pred, target):
        """
        pred and target should be of the same range as raw zernike data

        Parameters:
            pred: torch.Tensor, (b,t,c,h',w')
            target: torch.Tensor, (b,t,c,h',w')
        """
        pred = ZernikeDataset.zernike2phase(pred)
        target = ZernikeDataset.zernike2phase(target)
        return self.criterion(pred, target)
    


class PhaseMetric(nn.Module):
    """
    Metric for phase prediction
    """
    def __init__(self, **kwargs):
        """
        Initialize PhaseMetric

        Parameters:
        -----------
        kwargs: dict
            kwargs["dataset"]:
                test_size: list, [h,w], test_size for metric calculation
        """
        super(PhaseMetric, self).__init__()
        self.kwargs = kwargs
        
        self.test_size = kwargs["dataset"]["test_size"]
    
    def forward(self, pred, target, scalar=True):
        """
        pred and target should be of the same range as raw zernike data

        Parameters:
            pred: torch.Tensor, (b,t,c,h',w')
            target: torch.Tensor, (b,t,c,h',w')
            scalar: True return single scalar, otherwise return (ts,ts), here ts means test_size
        """
        pred = ZernikeDataset.zernike2phase(pred[:,-1,:,:,:].unsqueeze(1))
        target = ZernikeDataset.zernike2phase(target[:,-1,:,:,:].unsqueeze(1))
        b, t, p, h, w = pred.shape

        # stack over sliding window 
        R2_cum = torch.zeros(self.test_size[0], self.test_size[1], dtype=torch.float32, device=pred.device)
        rmse_cum = torch.zeros(self.test_size[0], self.test_size[1], dtype=torch.float32, device=pred.device)
        sigma_cum = torch.zeros(self.test_size[0], self.test_size[1], dtype=torch.float32, device=pred.device)
        N = 0

        for i in range(0, h-self.test_size[0]+1):
            for j in range(0, w-self.test_size[1]+1):
                temp_pred_phase = pred[:,:,:,i:i+self.test_size[0],j:j+self.test_size[1]]   # (b,t',p,ts,ts)
                temp_target_phase = target[:,:,:,i:i+self.test_size[0],j:j+self.test_size[1]]

                R2_cum = R2_cum + torch.mean(1.0 - torch.sum((temp_pred_phase - temp_target_phase)**2, dim=2) / torch.sum(temp_target_phase**2, dim=2), dim=(0,1))
                rmse_cum = rmse_cum + torch.mean(torch.std(temp_pred_phase - temp_target_phase, dim=2), dim=(0,1)) / 2 / torch.pi * 525
                sigma_cum = sigma_cum + torch.mean(torch.std(temp_pred_phase - temp_target_phase, dim=2) / (torch.amax(temp_target_phase, dim=2) - torch.amin(temp_target_phase, dim=2)), dim=(0,1))
                N += 1

        R2 = R2_cum / N
        rmse = rmse_cum / N
        sigma = sigma_cum / N

        # calculate metric
        ret_dict = {'R2':torch.tensor(-1), 'RMSE':torch.tensor(-1), 'SIGMA':torch.tensor(-1)}
        if scalar:
            ret_dict["R2"] = torch.mean(R2).item()
            ret_dict["RMSE"] = torch.mean(rmse).item()
            ret_dict["SIGMA"] = torch.mean(sigma).item()
        else:
            ret_dict['R2'] = R2
            ret_dict['RMSE'] = rmse
            ret_dict['SIGMA'] = sigma

        return ret_dict



class WarmupCosineAnnealingLR(optim.lr_scheduler.CosineAnnealingLR):
    """
    WarmupCosineAnnealingLR, used for learning rate scheduling by cosine annealing with warmup
    """
    def __init__(self, optimizer, batch_num, **kwargs):
        """
        Initialize WarmupCosineAnnealingLR

        Parameters:
        ------------
            optimizer: optimizer of the model
            batch_num: int, number of batches in an epoch
            kwargs: dict    
                kwargs["scheduler"]:
                    eta_min: float, minimum learning rate
                kwargs["config"]:
                    warmup_epoch: int, number of warmup epochs
                    virtual_epoch: int, number of virtual maximum epochs, used to calculate T_max
        """
        self.kwargs = kwargs
        self.eta_min = kwargs["scheduler"]["eta_min"]

        self.warmup_iters = kwargs['config']['warmup_epoch'] * batch_num
        self.T_max = kwargs['config']['virtual_epoch'] * batch_num
        
        super().__init__(optimizer, self.T_max, self.eta_min, -1)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [base_lr * (self.last_epoch + 1) / self.warmup_iters for base_lr in self.base_lrs]
        return super().get_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [base_lr * (self.last_epoch + 1) / self.warmup_iters for base_lr in self.base_lrs]
        return super()._get_closed_form_lr()
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['warmup_iters'] = self.warmup_iters
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        super().load_state_dict(state_dict)



