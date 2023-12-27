from helper import *
import scipy.io as io
from tqdm import tqdm
from matplotlib import pyplot as plt

# helper & device
Helper.random_seed(seed=42)
device = Helper.try_gpu()

def zernike_0406_stack():
    pred_data_path = ".\\data\\pred_data_2\\zernike0406\\"
    prep_data_path = ".\\data\\prep_data_230406\\"
    Helper.makedirs(pred_data_path)
    for set_id in range(28,49):
        zernike_data = torch.stack([torch.tensor(io.loadmat(prep_data_path + "%d\\zernike%d.mat"%(set_id, meta_id))['zernike']).type(torch.float32) for meta_id in range(200)], dim=0)
        print(zernike_data.shape)
        torch.save(zernike_data, pred_data_path + "zernike_data%d.pt"%(set_id))

def zernike_0407_stack():
    pred_data_path = ".\\data\\pred_data_new\\zernike0407_full\\"
    prep_data_path = ".\\data\\prep_data_230407_new_1\\"
    Helper.makedirs(pred_data_path)
    for set_id in range(2,4):
        zernike_data = torch.stack([torch.tensor(io.loadmat(prep_data_path + "%d\\zernike%d.mat"%(set_id, meta_id))['zernike']).type(torch.float32) for meta_id in range(1000)], dim=0)
        print(zernike_data.shape)
        torch.save(zernike_data, pred_data_path + "zernike_data%d.pt"%(set_id))  



if __name__ == "__main__":
    zernike_0407_stack()
    # zernike_0406_stack()

    print("Done!")