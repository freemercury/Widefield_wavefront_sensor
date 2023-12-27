from helper import *
from models import *
from matplotlib import pyplot as plt
import os

# helper & device
devices = Helper.try_all_gpus()
device = devices[0]

# path
BASE_PATH = os.getcwd() + "/"
REALIGN_DATA_PATH = BASE_PATH + "data/realign_data_230407/"
PREP_PATH = BASE_PATH + "data/prep_data_230407_new_2/"
PREP_PARAM_PATH = BASE_PATH + "data/prep_data_param/"
MASK_PATH = PREP_PARAM_PATH + "S2Zmatrix.mat"
MLP_PATH = PREP_PARAM_PATH + "mlp_model.pkl"

# phase config
SAVE_SHIFTMAP = False   # 存shiftmap
SAVE_ZERNIKE = True    # 存zernike
S_DEBUG = False # 存warped图和loss图
SET_START = 8
SET_STOP = 20
META_START = 40
META_STOP = 1039
TEST_VIEWS = [6,19,52,91,115,168,175,187,217]   # 测试视角，为0~224的index
NUM_VIEWS = 225
NUM_VIEW = 15
REF_VIEW_ID = 50    # 编号为0~224中的index
IMG_SIZE = [381,501]    # [height, width]
# training param
EPOCH1 = 20         # coarse training epoch, can be set to 0
EPOCH2 = 100         # fine training epoch, can be set to 0
NORM_TYPE = "avg"  # "gaussian" or "avg" or "none"
POOLING_SIZE = [19,25]    # avg pooling 
KERNEL_SIZE = 21   # gaussian kernel size, must be odd
SIGMA = 11               # gaussian kernel sigma
CTRL_SIZE = [19,25]
PROJ_TYPE = "mlp"   # "mlp" or "linear"
LR1 = 1e-2          # coarse training learning rate
LR2 = 1e-3          # fine training learning rate
LOSS_CROP_RATIO = 0.98
HIDDEN_SIZE = [300,500] # hidden layer size of mlp
NUM_ZERNIKE = 35     # number of zernike modes
SCALING_FACTOR = None   # set to None for automatic scaling, only used when proj_type="linear"
LOSS_TYPE = "l2"   # "l2", "l1", "grad", "ssim"



def img2zernike():
    """
    single gpu version
    """
    mask, P, valid_views = get_mask_P(MASK_PATH)
    model = PhaseMap(device, torch.zeros(NUM_VIEWS,1,IMG_SIZE[0],IMG_SIZE[1]), mask_path=MASK_PATH, ref_view=REF_VIEW_ID, 
                     norm_type=NORM_TYPE, pooling_size=POOLING_SIZE, kernel_size=KERNEL_SIZE, sigma=SIGMA,
                     ctrl_size=CTRL_SIZE, proj_type=PROJ_TYPE, loss_crop_ratio=LOSS_CROP_RATIO,
                     mlp_path=MLP_PATH, hidden_size=HIDDEN_SIZE, num_zernike=NUM_ZERNIKE, loss_type=LOSS_TYPE)
    for set_id in range(SET_START, SET_STOP+1):
        set_path = PREP_PATH + "%d/" % set_id
        Helper.makedirs(set_path)
        for meta_id in tqdm(range(META_START, META_STOP+1)):
            # load img
            img_path = REALIGN_DATA_PATH + "%d/realign/test_No%d.tif" % (set_id, meta_id)
            phantom = crop_meta_image(Helper.tif_read_img(img_path)).unsqueeze(1).to(device) # (225,1,340,340)
            # set model data
            model.set_phantom(phantom, REF_VIEW_ID)
            model.reset_shiftmap_models()
            # phase gen
            if SAVE_SHIFTMAP:
                # train model
                model.train(EPOCH1, LR1, EPOCH2, LR2)
                # save shiftmap
                io.savemat(set_path + "shiftmap%d.mat"%(meta_id-40), {"shiftmap": model.get_shiftmap(scaling=True).cpu().numpy()})
                # debug
                if S_DEBUG:
                    # save loss img
                    epoch_list, loss_list = model.get_loss_curve_shiftmap()
                    plt.figure()
                    plt.plot(epoch_list, loss_list)
                    plt.savefig(set_path + "shiftmap%d_loss.png"%(meta_id-40))
                    plt.close()
                    # save warped img
                    input_phantom, warped_phantom, target_phantom, input_norm, target_norm, test_views = model.get_warped_phantom(TEST_VIEWS)
                    for i, test_view in enumerate(test_views):
                        # save_img = torch.stack([input_phantom[i,:,:,:], input_norm[i,:,:,:], warped_phantom[i,:,:,:],
                        #                         target_phantom[i,:,:,:], target_norm[i,:,:,:]], dim=0).permute(0,2,3,1)
                        save_img = torch.stack([input_phantom[i,:,:,:], warped_phantom[i,:,:,:],
                                                target_phantom[i,:,:,:]], dim=0).permute(0,2,3,1)
                        # save_img = torch.stack([input_phantom[i,:,:,:], input_norm[i,:,:,:], input_phantom[i,:,:,:]/input_norm[i,:,:,:], warped_phantom[i,:,:,:],
                        #                         target_phantom[i,:,:,:], target_norm[i,:,:,:],target_phantom[i,:,:,:]/target_norm[i,:,:,:]], dim=0).permute(0,2,3,1)
                        Helper.tif_save_img(save_img, set_path + "shiftmap%d_view%d.tif"%(meta_id-40, test_view), save_type="float32")
            if SAVE_ZERNIKE:
                if SAVE_SHIFTMAP:
                    io.savemat(set_path + "zernike%d.mat"%(meta_id-40), {"zernike": model.get_zernike(scaling_factor=SCALING_FACTOR).detach().cpu().numpy()})
                else:
                    shiftmap_full = torch.tensor(io.loadmat(set_path + "shiftmap%d.mat"%(meta_id-40))["shiftmap"]).type(torch.float32).to(device)
                    shiftmap = shiftmap_convertor_fullsize2validviews(shiftmap_full, mask, valid_views)
                    io.savemat(set_path + "zernike%d.mat"%(meta_id-40), {"zernike": model.get_zernike(shiftmap=shiftmap, scaling_factor=SCALING_FACTOR).detach().cpu().numpy()})


if __name__ == "__main__":
    print("***********************\nPID: %d\n"%(os.getpid()))
    Helper.makedirs(PREP_PATH)
    Helper.makedirs(PREP_PARAM_PATH)

    img2zernike()


    print("Done!")
