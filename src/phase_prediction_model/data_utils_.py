from torch.utils.data import Dataset
from scipy.io import loadmat
import scipy.io as io


class ZernikeDataset(Dataset):
    train_data = []   # [(n1,c,h,w),(n2,c,h,w),...], all zernike
    train_list = []    # [(id1,id2)]
    test_data = []    # [(m1,c,h,w),(m2,c,h,w),...], all zernike
    test_list = []    # [(id1,id2)]
    mean = None   # (1,c,1,1) or (1,c,h,w) or 0 or [(1,c,h,w)] or [(1,c,1,1)], for zernike
    norm = None   # (1,1,c/p,1,1) or scalar, for zernike or phase
    Z2P, Z2p, p2Z = None, None, None
    
    def __init__(self, device, train=True, **kwargs):
        """
        device: 设备
        train: 是否为训练集

        data_path: "./data/pred_data_new/zernike0407/"
        data_param_path: "./data/pred_data_new/pred_data_param/"
        data_type: "zernike" or "phase"

        mean_type: "all", "patch", "none", "set_patch", "set_all"
        norm_type: "minimax", "std", "minimax_all", "std_all"
        norm_sets: 归一化数据使用范围, list
        phase_size: 计算loss采用phase的尺寸, int
        n_channel: 在0~34中选择需要的通道, list, [channel_start, channel_stop]

        train_sets: 训练集, list of int
        train_metas: list of int, 正数表示前n个meta, 负数表示后n个meta
        test_sets: 测试集, list of int
        test_metas: list of int, 正数表示前n个meta, 负数表示后n个meta

        t_series: 时间序列长度, input总长度
        t_offset: 时间序列偏移量, =0表示输入最后一个时刻为带预测时刻
        t_down: 时间序列下采样间隔, 表示时序抽点间隔, 默认为0即不抽点
        all_patch: 全区域, (h_start, h_end, w_start, w_end)
        input_patch: 输入区域, (h_start, h_end, w_start, w_end)
        target_patch: 输出区域, (h_start, h_end, w_start, w_end)
        test_size: 每次测试区域的大小, (h_size, w_size)
        """
        super().__init__()

        self.device = device
        self.train = train
        self.kwargs = kwargs

        self.data_path = self.kwargs["dataset"]["data_path"]
        self.data_param_path = self.kwargs["dataset"]["data_param_path"]
        self.data_type = self.kwargs["dataset"]["data_type"]

        self.mean_type = self.kwargs["dataset"]["mean_type"]
        self.norm_type = self.kwargs["dataset"]["norm_type"]
        self.norm_sets = self.kwargs["dataset"]["norm_sets"]
        self.phase_size = self.kwargs["dataset"]["phase_size"]
        self.n_channel = list(range(self.kwargs["dataset"]["n_channel"][0], self.kwargs["dataset"]["n_channel"][1]+1))

        self.train_sets = self.kwargs["dataset"]["train_sets"]
        self.train_metas = self.kwargs["dataset"]["train_metas"]
        self.test_sets = self.kwargs["dataset"]["test_sets"]
        self.test_metas = self.kwargs["dataset"]["test_metas"]

        self.t_series = self.kwargs["dataset"]["t_series"]
        self.t_offset = self.kwargs["dataset"]["t_offset"]
        self.t_down = self.kwargs["dataset"]["t_down"]
        self.all_patch = self.kwargs["dataset"]["all_patch"]
        self.input_patch = self.kwargs["dataset"]["input_patch"]
        self.target_patch = self.kwargs["dataset"]["target_patch"]
        self.test_size = self.kwargs["dataset"]["test_size"]

        self.seq_len = (self.t_series + self.t_offset) + (self.t_series + self.t_offset - 1) * self.t_down

        if ZernikeDataset.norm is None:
            ZernikeDataset.load_zernike_phase(self.device, self.data_param_path, self.phase_size, self.n_channel)
            ZernikeDataset.calc_norm_zernike(self.device, self.norm_sets, self.data_path, self.data_type, self.mean_type, self.norm_type, self.n_channel)
            ZernikeDataset.update(self.device, self.data_path, self.mean_type, 
                                  self.train_sets, self.train_metas, self.test_sets, self.test_metas, 
                                  self.norm_sets, self.n_channel, self.seq_len)

    def __len__(self):
        return len(ZernikeDataset.train_list) if self.train else len(ZernikeDataset.test_list)

    def __getitem__(self, id):
        if self.train:
            id1, id2 = ZernikeDataset.train_list[id]
            set_id = self.train_sets[id1]
            metas = self.train_metas[id1]
            input_id_index = list(range(id2, id2 + self.seq_len - self.t_offset * (self.t_down + 1), self.t_down + 1))
            target_id_index = list(range(id2 + self.t_offset * (self.t_down + 1), id2 + self.seq_len, self.t_down + 1))
            if metas > 0:
                input_meta_index, target_meta_index = input_id_index, target_id_index
            else:
                input_meta_index = [v + 1000 + metas for v in input_id_index]
                target_meta_index = [v + 1000 + metas for v in target_id_index]
            
            input = ZernikeDataset.train_data[id1][input_id_index,:,self.input_patch[0]:self.input_patch[1],self.input_patch[2]:self.input_patch[3]].clone()  # (t,c,h,w)
            if self.t_offset == 0:
                crop_patch = [self.target_patch[0] - self.input_patch[0], self.target_patch[1] - self.input_patch[0], 
                              self.target_patch[2] - self.input_patch[2], self.target_patch[3] - self.input_patch[2]]
                input[:, :, crop_patch[0]:crop_patch[1], crop_patch[2]:crop_patch[3]] = 0.0
            target = ZernikeDataset.train_data[id1][target_id_index,:,self.target_patch[0]:self.target_patch[1],self.target_patch[2]:self.target_patch[3]].clone()
            if self.data_type == "zernike":
                input = self.normalize(input.unsqueeze(0)).squeeze(0)
                target = self.normalize(target.unsqueeze(0)).squeeze(0)
            elif self.data_type == "phase":
                input = self.normalize(self.zernike2phase(input.unsqueeze(0))).squeeze(0)
                target = self.normalize(self.zernike2phase(target.unsqueeze(0))).squeeze(0)

        else:
            id1, id2 = ZernikeDataset.test_list[id]
            set_id = self.test_sets[id1]
            metas = self.test_metas[id1]
            input_id_index = list(range(id2, id2 + self.seq_len - self.t_offset * (self.t_down + 1), self.t_down + 1))
            target_id_index = list(range(id2 + self.t_offset * (self.t_down + 1), id2 + self.seq_len, self.t_down + 1))
            if metas > 0:
                input_meta_index, target_meta_index = input_id_index, target_id_index
            else:
                input_meta_index = [v + 1000 + metas for v in input_id_index]
                target_meta_index = [v + 1000 + metas for v in target_id_index]

            input = ZernikeDataset.test_data[id1][input_id_index,:,self.input_patch[0]:self.input_patch[1],self.input_patch[2]:self.input_patch[3]].clone()  # (t,c,h,w)
            if self.t_offset == 0:
                crop_patch = [self.target_patch[0] - self.input_patch[0], self.target_patch[1] - self.input_patch[0], 
                              self.target_patch[2] - self.input_patch[2], self.target_patch[3] - self.input_patch[2]]
                input[:, :, crop_patch[0]:crop_patch[1], crop_patch[2]:crop_patch[3]] = 0.0
            target = ZernikeDataset.test_data[id1][target_id_index,:,self.target_patch[0]:self.target_patch[1],self.target_patch[2]:self.target_patch[3]].clone()
            if self.data_type == "zernike":
                input = self.normalize(input.unsqueeze(0)).squeeze(0)
                target = self.normalize(target.unsqueeze(0)).squeeze(0)
            elif self.data_type == "phase":
                input = self.normalize(self.zernike2phase(input.unsqueeze(0))).squeeze(0)
                target = self.normalize(self.zernike2phase(target.unsqueeze(0))).squeeze(0)

        return input, target, (set_id, input_meta_index, target_meta_index)

    @classmethod
    def reset(cls):
        cls.train_data = []
        cls.train_list = []
        cls.test_data = []
        cls.test_list = []
        cls.mean = None
        cls.norm = None
        cls.Z2P, cls.Z2p, cls.p2Z = None, None, None

    @classmethod
    def load_zernike_phase(cls, device, data_param_path, phase_size, n_channel):
        """
        load Z2P, Z2p, p2Z
        """
        trans_dict = loadmat(data_param_path + "zernike_phase%d.mat" % (phase_size))
        cls.Z2P = torch.from_numpy(trans_dict["Z2P"]).type(torch.float32).to(device)[n_channel,:,:]
        cls.Z2p = torch.from_numpy(trans_dict["Z2p"]).type(torch.float32).to(device)[n_channel,:]
        cls.p2Z = torch.from_numpy(trans_dict["p2Z"]).type(torch.float32).to(device)[:,n_channel]

    @classmethod
    def zernike2phase(cls, zernike_data):
        """
        zernike_data: (b,t,c,h,w)
        return: (b,t,p,h,w)
        """
        phase_data = F.linear(zernike_data.permute(0,1,3,4,2), cls.Z2p.permute(1,0), bias=None).permute(0,1,4,2,3)
        return phase_data
    
    @classmethod
    def phase2zernike(cls, phase_data):
        """norm_sets
        phase_data: (b,t,p,h,w)
        return: (b,t,c,h,w)
        """
        zernike_data = F.linear(phase_data.permute(0,1,3,4,2), cls.p2Z.permute(1,0), bias=None).permute(0,1,4,2,3)
        return zernike_data
    
    @classmethod
    def calc_norm_zernike(cls, device, norm_sets, data_path, data_type, mean_type, norm_type, n_channel):
        data = []
        for set_id in norm_sets:
            data.append(torch.load(data_path + "zernike_data%d.pt"%(set_id)).type(torch.float32).to(device)[:,n_channel,:,:])
        data = torch.stack(data, dim=0)   # (n,1000,c,h,w)

        # calc cls.mean
        if mean_type == "all":
            cls.mean = torch.mean(data, dim=(0,1,3,4), keepdim=True).squeeze(0)
        elif mean_type == "patch":
            cls.mean = torch.mean(data, dim=(0,1), keepdim=True).squeeze(0)
        elif mean_type == "none":
            cls.mean = 0.0 * torch.mean(data, dim=(0,1,3,4), keepdim=True).squeeze(0)
        elif mean_type == "set_all":
            mean = torch.mean(data, dim=(1,3,4), keepdim=True)  # (n,1,c,1,1)
            cls.mean = [mean[i,:,:,:,:] for i in range(mean.shape[0])]
        elif mean_type == "set_patch":
            mean = torch.mean(data, dim=(1), keepdim=True)  # (n,1,c,h,w)
            cls.mean = [mean[i,:,:,:,:] for i in range(mean.shape[0])]

        # subtract cls.mean
        if mean_type in ["all", "patch", "none"]:
            data = data - cls.mean
        elif mean_type in ["set_all", "set_patch"]:
            data = data - torch.cat(cls.mean, dim=0)
        
        # calc cls.norm
        if data_type == "phase":
            data = cls.zernike2phase(data)
        if norm_type == "minimax":
            cls.norm = torch.amax(torch.abs(data), dim=(0,1,3,4), keepdim=True)
        elif norm_type == "std":
            cls.norm = torch.std(data, dim=(0,1,3,4), keepdim=True)
        elif norm_type == "minimax_all":
            cls.norm = torch.amax(torch.abs(data)).item()
        elif norm_type == "std_all":
            cls.norm = torch.std(data).item()

    @classmethod
    def normalize(cls, data):
        """
        data: (b,t,c/p,h,w)
        """
        return data / cls.norm
    
    @classmethod
    def denormalize(cls, data):
        """
        data: (b,t,c/p,h,w)
        """
        return data * cls.norm
    
    @classmethod
    def update(cls, device, data_path, mean_type, train_sets, train_metas, test_sets, test_metas, norm_sets, n_channel, seq_len):
        # update cls.train_data and cls.train_list
        for set_id, meta_len in zip(train_sets, train_metas):
            temp_data = torch.load(data_path + "zernike_data%d.pt"%(set_id)).type(torch.float32).to(device)[:,n_channel,:,:] # (n,c,h,w)
            if meta_len > 0:
                temp_data = temp_data[:meta_len,:,:,:]
            else:
                temp_data = temp_data[meta_len:,:,:,:]
            if mean_type in ["set_all", "set_patch"]:
                temp_data = temp_data - cls.mean[norm_sets.index(set_id)]
            else:
                temp_data = temp_data - cls.mean
            cls.train_data.append(temp_data)
            cls.train_list += [(train_sets.index(set_id), i) for i in range(abs(meta_len) - seq_len + 1)]
        
        # update cls.test_data and cls.test_list
        for set_id, meta_len in zip(test_sets, test_metas):
            temp_data = torch.load(data_path + "zernike_data%d.pt"%(set_id)).type(torch.float32).to(device)[:,n_channel,:,:] # (n,c,h,w)
            if meta_len > 0:
                temp_data = temp_data[:meta_len,:,:,:]
            else:
                temp_data = temp_data[meta_len:,:,:,:]
            if mean_type in ["set_all", "set_patch"]:
                temp_data = temp_data - cls.mean[norm_sets.index(set_id)]
            else:
                temp_data = temp_data - cls.mean
            cls.test_data.append(temp_data)
            cls.test_list += [(test_sets.index(set_id), i) for i in range(abs(meta_len) - seq_len + 1)]
    