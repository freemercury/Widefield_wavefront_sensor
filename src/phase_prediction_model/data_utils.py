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

    






if __name__ == "__main__":

    print("Done!")