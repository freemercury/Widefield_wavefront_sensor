from helper import *
import time
from models import *
from matplotlib import pyplot as plt
import os
import multiprocessing as mp
import subprocess as sp

# helper & device
devices = Helper.try_all_gpus()
device = devices[0]

# multi processing
mp.set_start_method('forkserver', force=True)


# path
MASK_PATH = "./S2Zmatrix.mat"
MLP_PATH = "./mlp_model.pkl"

# phase config
SAVE_SHIFTMAP = True   # 存shiftmap
SAVE_ZERNIKE = True    # 存zernike
S_DEBUG = False # 存warped图和loss图
SET_START = 1
SET_STOP = 7
META_START = 40
META_STOP = 1039
TEST_VIEWS = [6,19,52,91,112,168,175,187,217]   # 测试视角，为0~224的index
NUM_VIEWS = 225
NUM_VIEW = 15
REF_VIEW_ID = 50    # 编号为0~224中的index
IMG_SIZE = [381,501]    # [height, width]
# training param
EPOCH1 = 20         # coarse training epoch, can be set to 0
EPOCH2 = 10         # fine training epoch, can be set to 0
NORM_TYPE = "gaussian"  # "gaussian" or "avg" or "none"
POOLING_SIZE = [3,4]    # avg pooling 
KERNEL_SIZE = 21   # gaussian kernel size, must be odd
SIGMA = 15               # gaussian kernel sigma
CTRL_SIZE = [17,17]
PROJ_TYPE = "mlp"   # "mlp" or "linear"
LR1 = 1e-2          # coarse training learning rate
LR2 = 1e-3          # fine training learning rate
LOSS_CROP_RATIO = 0.98
HIDDEN_SIZE = [300,500] # hidden layer size of mlp
NUM_ZERNIKE =35     # number of zernike modes
SCALING_FACTOR = None   # set to None for automatic scaling, only used when proj_type="linear"
LOSS_TYPE = "l2"   # "l2", "l1", "grad", "ssim"
# multigpu speed test config
TOTAL_LENGTH = 3500
CUDA_CNT = 7


def sub_train(i):
    """
    single gpu process for multi gpu version
    """
    dummy_phantom = torch.randn(225,1,340,340)
    model = PhaseMap(device, torch.zeros(NUM_VIEWS,1,IMG_SIZE[0],IMG_SIZE[1]), mask_path=MASK_PATH, ref_view=REF_VIEW_ID, 
                     norm_type=NORM_TYPE, pooling_size=POOLING_SIZE, kernel_size=KERNEL_SIZE, sigma=SIGMA,
                     ctrl_size=CTRL_SIZE, proj_type=PROJ_TYPE, loss_crop_ratio=LOSS_CROP_RATIO,
                     mlp_path=MLP_PATH, hidden_size=HIDDEN_SIZE, num_zernike=NUM_ZERNIKE, loss_type=LOSS_TYPE)
    for _ in range(TOTAL_LENGTH // CUDA_CNT):
        model.set_phantom(dummy_phantom, REF_VIEW_ID)
        model.reset_shiftmap_models()
        model.train(EPOCH1, LR1, EPOCH2, LR2)
        zernike = model.get_zernike(scaling_factor=SCALING_FACTOR)
    
def shiftmap_mp():
    """
    pseduo multi gpu version, just for speed test
    """
    processes = []
    t0 = time.time()
    for i in range(CUDA_CNT):
        p = mp.Process(target=sub_train, args=(i,))
        processes.append(p)
        p.start()
    for i in range(CUDA_CNT):
        processes[i].join()
    t1 = time.time()
    print("Average time: %f"%((t1-t0)/TOTAL_LENGTH*CUDA_CNT/8))

if __name__ == "__main__":
    print("***********************\nPID: %d\n"%(os.getpid()))
    shiftmap_mp()

    print("Done!")
