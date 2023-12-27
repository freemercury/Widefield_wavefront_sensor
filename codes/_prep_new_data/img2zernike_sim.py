from helper import *
from models import *
from matplotlib import pyplot as plt
import os

# helper & device
devices = Helper.try_all_gpus()
device = devices[0]

# path
BASE_PATH = os.getcwd() + "/"
REALIGN_DATA_PATH = BASE_PATH + "data/sim_data/"
PREP_PATH = BASE_PATH + "data/sim_prep_data/"
PREP_PARAM_PATH = BASE_PATH + "data/prep_data_param/"
MASK_PATH = PREP_PARAM_PATH + "64_S2Zmatrix.mat"
MLP_PATH = None

# phase config
SAVE_SHIFTMAP = True   # 存shiftmap
S_DEBUG = True # 存warped图和loss图
TEST_VIEWS = [2,5,9,11,14,16,19,21,23,24,27,31,32,36,40,43,49,61]   # 测试视角，为0~224的index
NUM_VIEWS = 64
NUM_VIEW = 8
REF_VIEW_ID = 36    # 编号为0~224中的index
IMG_SIZE = [360,450]    # [height, width]
# training param
EPOCH1 = 20         # coarse training epoch, can be set to 0
EPOCH2 = 100         # fine training epoch, can be set to 0
NORM_TYPE = "none"  # "gaussian" or "avg" or "none"
POOLING_SIZE = [1,1]    # avg pooling 
KERNEL_SIZE = 21   # gaussian kernel size, must be odd
SIGMA = 11               # gaussian kernel sigma
CTRL_SIZE = [6,7]
PROJ_TYPE = "mlp"   # "mlp" or "linear"
LR1 = 1e-4          # coarse training learning rate
LR2 = 1e-4          # fine training learning rate
LOSS_CROP_RATIO = 1.0
HIDDEN_SIZE = [300,500] # hidden layer size of mlp
NUM_ZERNIKE = 35     # number of zernike modes
SCALING_FACTOR = None   # set to None for automatic scaling, only used when proj_type="linear"
LOSS_TYPE = "ssim"    # "l2", "l1", "grad", "ssim"



def img2zernike():
    """
    single gpu version
    """
    mask, P, valid_views = get_mask_P(MASK_PATH)
    model = PhaseMap(device, torch.zeros(NUM_VIEWS,1,IMG_SIZE[0],IMG_SIZE[1]), mask_path=MASK_PATH, ref_view=REF_VIEW_ID, 
                     norm_type=NORM_TYPE, pooling_size=POOLING_SIZE, kernel_size=KERNEL_SIZE, sigma=SIGMA,
                     ctrl_size=CTRL_SIZE, proj_type=PROJ_TYPE, loss_crop_ratio=LOSS_CROP_RATIO,
                     mlp_path=MLP_PATH, hidden_size=HIDDEN_SIZE, num_zernike=NUM_ZERNIKE, loss_type=LOSS_TYPE)
    img_path = REALIGN_DATA_PATH + "realign/1.tif"
    phantom = crop_meta_image(Helper.tif_read_img(img_path)).unsqueeze(1).to(device) # (64,1,340,340)
    # set model data
    model.set_phantom(phantom, REF_VIEW_ID)
    model.reset_shiftmap_models()
    # phase gen
    if SAVE_SHIFTMAP:
        # train model
        model.train(EPOCH1, LR1, EPOCH2, LR2)
        # save shiftmap
        io.savemat(PREP_PATH + "shiftmap.mat", {"shiftmap": model.get_shiftmap(scaling=True).cpu().numpy()})
        # debug
        if S_DEBUG:
            # save loss img
            epoch_list, loss_list = model.get_loss_curve_shiftmap()
            plt.figure()
            plt.plot(epoch_list, loss_list)
            plt.savefig(PREP_PATH + "shiftmap_loss.png")
            plt.close()
            # save warped img
            input_phantom, warped_phantom, target_phantom, input_norm, target_norm, test_views = model.get_warped_phantom(TEST_VIEWS)
            for i, test_view in enumerate(test_views):
                # save_img = torch.stack([input_phantom[i,:,:,:], input_norm[i,:,:,:], warped_phantom[i,:,:,:],
                #                         target_phantom[i,:,:,:], target_norm[i,:,:,:]], dim=0).permute(0,2,3,1)
                save_img = torch.stack([input_phantom[i,:,:,:], warped_phantom[i,:,:,:],
                                        target_phantom[i,:,:,:]], dim=0).permute(0,2,3,1)
                Helper.tif_save_img(save_img, PREP_PATH + "shiftmap_view%d.tif"%(test_view), save_type="float32")



if __name__ == "__main__":
    print("***********************\nPID: %d\n"%(os.getpid()))
    Helper.makedirs(PREP_PATH)
    Helper.makedirs(PREP_PARAM_PATH)

    img2zernike()

    print("Done!")
