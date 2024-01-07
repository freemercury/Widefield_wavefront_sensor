from torch.utils.data import Dataset
from scipy.io import loadmat
import scipy.io as ios

class ZernikeDataset(Dataset):
    train_data = []   # [(n1,c,h,w),(n2,c,h,w),...], all zernike
    train_list = []    # [(id1,id2)]
    test_data = []    # [(m1,c,h,w),(m2,c,h,w),...], all zernike
    test_list = []    # [(id1,id2)]
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

        self.sets = self.kwargs["dataset"]["sets"]
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
            ZernikeDataset.load_zernike_phase(self.device, self.data_param_path, self.phase_size, self.n_channel)
            ZernikeDataset.calc_norm_zernike(self.device, self.norm_sets, self.data_path, self.data_type, self.mean_type, self.norm_type, self.n_channel)
            ZernikeDataset.update(self.device, self.data_path, self.mean_type, 
                                  self.train_sets, self.train_metas, self.test_sets, self.test_metas, 
                                  self.norm_sets, self.n_channel, self.seq_len)






if __name__ == "__main__":

    print("Done!")