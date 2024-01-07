from torch.utils.data import Dataset
from scipy.io import loadmat
import scipy.io as ios
import torch
import numpy as np
import os, re, glob

class ZernikeDataset(Dataset):
    data_list = []   # [(n1,c,h,w),(n2,c,h,w),...], all zernike
    file_list = []  # [file_list1, file_list2, ...], all files
    train_id_list = []    # [(id1,id2)]
    test_id_list = []    # [(id1,id2)]
    mean = None   # (1,1,c,h,w), system aberration
    norm = None   # scalar
    Z2P, Z2p, p2Z = None, None, None

    def __init__(self, device, train=True, **kwargs):
        super().__init__()

        self.device = device
        self.train = train
        self.kwargs = kwargs

        self.data_path = self.kwargs["dataset"]["data_path"]
        self.zernike_phase_path = self.kwargs["dataset"]["data_param_path"]

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

    @classmethod
    def reset(cls):
        cls.data_list = []
        cls.file_list = []
        cls.train_id_list = []
        cls.test_id_list = []
        cls.mean = None
        cls.norm = None
        cls.Z2P, cls.Z2p, cls.p2Z = None, None, None

    @classmethod
    def load_zernike_phase(cls, device, zernike_phase_path, phase_size, n_channel):
        """
        load Z2P, Z2p, p2Z
        """
        trans_dict = loadmat(zernike_phase_path + "zernike_phase%d.mat" % (phase_size))
        cls.Z2P = torch.from_numpy(trans_dict["Z2P"]).type(torch.float32).to(device)[n_channel,:,:]
        cls.Z2p = torch.from_numpy(trans_dict["Z2p"]).type(torch.float32).to(device)[n_channel,:]
        cls.p2Z = torch.from_numpy(trans_dict["p2Z"]).type(torch.float32).to(device)[:,n_channel]
    
    @classmethod
    def load_data(cls, device, data_path, n_channel, seq_len, split):
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
            set_path = data_path + set_name + "/"
            file_list = glob.glob(set_path + "*_mlp_zernike.mat")
            file_list.sort(key=lambda x: extract_number(x.split("\\")[-1], r'.*?(\d+).*?'))
            tmp_data = torch.stack([torch.from_numpy(loadmat(file)["zernike"]).type(torch.float32).to(device)[n_channel,:,:] for file in file_list], dim=0)
            cls.data_list.append(tmp_data)
            cls.file_list.append(file_list)
            train_len = int(len(file_list) * split)
            cls.train_id_list += [(set_id, i) for i in range(train_len - seq_len + 1)]
            cls.test_id_list += [(set_id, i) for i in range(train_len - seq_len + 1, len(file_list) - seq_len + 1)]



if __name__ == "__main__":
    data_path = "./data/fake/set1/"
    # 首先找到data_path下边一级的所有set文件夹
    set_list = os.listdir(data_path)
    def extract_number(filename, pattern):
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        return 0

    # set_list.sort(key=extract_number)
    # print(set_list)
    # print(list(map(extract_number, set_list)))
    # filename = "test_No121"
    # match = re.search(r'.*?(\d+).*?', filename)
    # print(match.group(1))
    tmp = glob.glob(data_path + "*_mlp_zernike.txt")
    print(tmp)
    print(list(map(lambda x: extract_number(x.split("\\")[-1], r'.*?(\d+).*?'), tmp)))



    print("Done!")