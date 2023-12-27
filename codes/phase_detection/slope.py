from helper import *
from slope_estimation_model import *
import argparse
import glob
from scipy import io
from tqdm import tqdm
from matplotlib import pyplot as plt


REALIGN_DATA_PATH = "./data/realign_data/230407/set1/"
PHASE_DATA_PATH = "./data/phase_data/230407/set1/"
MASK_PATH = "./data/settings/mask.mat"
GPU_ID = 0  # set to None for cpu
DEBUG = False # whether to save warped images and loss curve
TEST_VIEWS = [6,19,52,91,115,168,175,187,217]   # index in [0, n_view_x*n_view_y-1]
MASK_SIZE = [15,15]
IMG_SIZE = [381,501]
REF_VIEW_ID = 50    # index in [0, n_view_x*n_view_y-1]
EPOCHS = [20,100]
LRS = [1e-2,1e-3]
NORM_TYPE = "avg"  # "gauss" or "avg" or None
POOLING_SIZE = [19,25]    # avg pooling
KERNEL_SIZE = 21   # gaussian kernel size, must be odd
SIGMA = 11               # gaussian kernel sigma
CTRL_SIZE = [19,25]
LOSS_CROP_RATIO = 0.98
LOSS_TYPE = "l2"   # "l2", "l1", "grad"



# args
def parse_args():
    parser = argparse.ArgumentParser(description='Slope Estimation Arguments')
    parser.add_argument('--realign_data_path', default=REALIGN_DATA_PATH, type=str, help='realign data path')
    parser.add_argument('--phase_data_path', default=PHASE_DATA_PATH, type=str, help='phase data path')
    parser.add_argument('--mask_path', default=MASK_PATH, type=str, help='mask path')
    parser.add_argument('--gpu_id', default=GPU_ID, type=int, help='gpu id, -1 for cpu')
    parser.add_argument('--debug', default=DEBUG, type=bool, help='whether to save warped images and loss curve')
    parser.add_argument('--test_views', default=TEST_VIEWS, type=str, help='test views')
    parser.add_argument('--mask_size', default=MASK_SIZE, type=str, help='mask size (h,w)')
    parser.add_argument('--img_size', default=IMG_SIZE, type=str, help='image size (h,w)')
    parser.add_argument('--ref_view_id', default=REF_VIEW_ID, type=int, help='reference view id')
    parser.add_argument('--epochs', default=EPOCHS, type=str, help='two epochs for coarse and fine training')
    parser.add_argument('--lrs', default=LRS, type=str, help='two learning rates for coarse and fine training')
    parser.add_argument('--norm_type', default=NORM_TYPE, type=str, help='norm type, "gauss" or "avg" or None')
    parser.add_argument('--pooling_size', default=POOLING_SIZE, type=str, help='pooling size (h,w)')
    parser.add_argument('--kernel_size', default=KERNEL_SIZE, type=int, help='gaussian kernel size, must be odd')
    parser.add_argument('--sigma', default=SIGMA, type=int, help='gaussian kernel sigma')
    parser.add_argument('--ctrl_size', default=CTRL_SIZE, type=str, help='control points size (m,n)')
    parser.add_argument('--loss_crop_ratio', default=LOSS_CROP_RATIO, type=float, help='loss crop ratio')
    parser.add_argument('--loss_type', default=LOSS_TYPE, type=str, help='loss type, "l2", "l1", "grad"')
    args = parser.parse_args()
    if isinstance(args.test_views, str):
        args.test_views = [int(i) for i in args.test_views.split(',')]
    if isinstance(args.mask_size, str):
        args.mask_size = [int(i) for i in args.mask_size.split(',')]
    if isinstance(args.img_size, str):
        args.img_size = [int(i) for i in args.img_size.split(',')]
    if isinstance(args.epochs, str):
        args.epochs = [int(i) for i in args.epochs.split(',')]
    if isinstance(args.lrs, str):
        args.lrs = [float(i) for i in args.lrs.split(',')]
    if isinstance(args.pooling_size, str):
        args.pooling_size = [int(i) for i in args.pooling_size.split(',')]
    if isinstance(args.ctrl_size, str):
        args.ctrl_size = [int(i) for i in args.ctrl_size.split(',')]
    return vars(args)




def img2slope():
    args = parse_args()

    # set device
    if args["gpu_id"] == -1:
        device = Helper.get_cpu()
    else:
        device = Helper.try_gpu(device_id=args["gpu_id"])

    # random seed
    Helper.random_seed()

    # files
    if not op.exists(args["mask_path"]):
        args["mask_path"] = None
    if not op.exists(args["realign_data_path"]):
        raise Exception("realign data path not exist!")
    Helper.makedirs(args["phase_data_path"])
    Helper.save_json(args, args["phase_data_path"] + "args_" + Helper.get_timestamp() + ".json")
    realign_data_files = glob.glob(args["realign_data_path"] + "*.tif")

    # initialize model
    model = SlopeEstimation(device=device,
                            mask_size=args["mask_size"],
                            phantom_size=args["img_size"],
                            phantom=None,
                            mask_path=args["mask_path"],
                            ref_view=args["ref_view_id"],
                            norm_type=args["norm_type"],
                            pooling_size=args["pooling_size"],
                            kernel_size=args["kernel_size"],
                            sigma=args["sigma"],
                            ctrl_size=args["ctrl_size"],
                            loss_type=args["loss_type"],
                            loss_crop_ratio=args["loss_crop_ratio"],
                            epochs=args["epochs"],
                            lrs=args["lrs"])

    # img2slope
    for file in tqdm(realign_data_files):
        phantom = Helper.tif_read_img(file).unsqueeze(1).to(device)
        model.set_phantom(phantom, args["ref_view_id"])
        model.reset_shiftmap_models()
        model.train()
        io.savemat(args["phase_data_path"] + file.split("\\")[-1].replace(".tif", "_slope.mat"), 
                   {"slope": model.get_shiftmap(scaling=True).cpu().numpy()})
        
        # debug
        if args["debug"]:
            # save loss img
            epoch_list, loss_list = model.get_loss_curve_shiftmap()
            plt.figure()
            plt.plot(epoch_list, loss_list)
            plt.savefig(args["phase_data_path"] + file.split("\\")[-1].replace(".tif", "_slope_loss.png"))
            plt.close()
            # save warped img
            input_phantom, warped_phantom, target_phantom, _, __build_class__, test_views = model.get_warped_phantom(args["test_views"])
            for i, test_view in enumerate(test_views):
                save_img = torch.stack([input_phantom[i,:,:,:], warped_phantom[i,:,:,:],
                                        target_phantom[i,:,:,:]], dim=0).permute(0,2,3,1)
                Helper.tif_save_img(save_img, 
                                    args["phase_data_path"] + file.split("\\")[-1].replace(".tif", "_slope_view%d.png"%(test_view)),
                                    save_type="float32")


if __name__ == "__main__":
    img2slope()