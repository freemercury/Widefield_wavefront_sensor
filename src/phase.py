from helper import *
from phase_detection_model import PhaseProjection
import argparse
from scipy import io
import glob
from tqdm import tqdm

DATA_PATH = "./data/phase_data/230406/set1/"
CKPT_PATH = "./log/mlp/"
MASK_PATH = "./data/settings/mask.mat"
PHASE_ZERNIKE_PATH = "./data/settings/zernike_phase55.mat"
PHASE_SIZE = 55
JOB = "infer"  # "train" or "test" or "infer" or "remove_sys" or "draw"
GPU_ID = 0  # set to None for cpu
MASK_SIZE = [15,15]
HIDDEN_SIZE = [300,500] # hidden layer size of mlp
NUM_ZERNIKE = 35     # number of zernike modes
EPOCH = 20  # number of epochs
LR = 1e-4   # learning rate
BATCH_SIZE = 128    # batch size
SPLIT = 0.9 # train / (train + valid) split ratio

# args
def parse_args():
    parser = argparse.ArgumentParser(description='Phase Projection Arguments')
    parser.add_argument('--data_path', default=DATA_PATH, type=str, help='data path')
    parser.add_argument('--job', default=JOB, type=str, help='job type, "train" or "test" or "infer" or "remove_sys" or "draw"')
    parser.add_argument('--ckpt_path', default=CKPT_PATH, type=str, help='checkpoint path')
    parser.add_argument('--mask_path', default=MASK_PATH, type=str, help='mask path')
    parser.add_argument('--phase_zernike_path', default=PHASE_ZERNIKE_PATH, type=str, help='phase zernike path')
    parser.add_argument('--phase_size', default=PHASE_SIZE, type=int, help='phase size')
    parser.add_argument('--gpu_id', default=GPU_ID, type=int, help='gpu id, -1 for cpu')
    parser.add_argument('--mask_size', default=MASK_SIZE, type=str, help='mask size (h,w)')
    parser.add_argument('--hidden_size', default=HIDDEN_SIZE, type=str, help='hidden layer size of mlp')
    parser.add_argument('--num_zernike', default=NUM_ZERNIKE, type=int, help='number of zernike modes')
    parser.add_argument('--epoch', default=EPOCH, type=int, help='number of epochs')
    parser.add_argument('--lr', default=LR, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')
    parser.add_argument('--valid', action='store_true', help='use validation set, default is False')
    parser.add_argument('--split', default=SPLIT, type=float, help='train / (train + valid) split ratio')
    args = parser.parse_args()
    if isinstance(args.mask_size, str):
        args.mask_size = [int(i) for i in args.mask_size.split(',')]
    if isinstance(args.hidden_size, str):
        args.hidden_size = [int(i) for i in args.hidden_size.split(',')]
    return vars(args)


def slope2zernike():
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
    if not op.exists(args["data_path"]):
        raise Exception("data path not exist!")
    Helper.makedirs("./log/args/")
    Helper.save_json(args, "./log/args/phase_args_" + Helper.get_timestamp() + ".json")
    Helper.makedirs(args["ckpt_path"])

    # initialize model
    model = PhaseProjection(device=device,
                            hidden_size=args["hidden_size"],
                            num_zernike=args["num_zernike"],
                            mask_size=args["mask_size"],
                            mask_path=args["mask_path"],
                            epoch=args["epoch"],
                            lr=args["lr"],
                            batch_size=args["batch_size"],
                            valid=args["valid"],
                            split=args["split"])
    
    # main process
    if args["job"] == "train":
        model.train(args["data_path"])
        model.save(args["ckpt_path"])
    elif args["job"] == "test":
        model.load(args["ckpt_path"], epoch=args["epoch"] - 1)
        model.test(args["data_path"])
    elif args["job"] == "infer":
        model.load(args["ckpt_path"], epoch=args["epoch"] - 1)
        model.inference(args["data_path"])
    elif args["job"] == "remove_sys":
        # load data
        file_list = glob.glob(args["data_path"] + "/**/*_mlp_zernike.mat", recursive=True)
        # file_list1 = glob.glob(args["data_path"] + "/**/*_zernike.mat", recursive=True)
        # file_list = []
        # for filename in file_list1:
        #     if "mlp" in filename:
        #         file_list.append(filename)
        #     elif filename.replace("_zernike.mat", "_mlp_zernike.mat") not in file_list1:
        #         file_list.append(filename)
        all_zernike = torch.stack([torch.from_numpy(io.loadmat(file)["zernike"]).type(torch.float32) for file in file_list], dim=0) # (n,c,h,w)
        mean_zernike = torch.mean(all_zernike, dim=0) # (c,h,w)
        for filename in file_list:
            zernike = torch.from_numpy(io.loadmat(filename)["zernike"]).type(torch.float32)
            zernike -= mean_zernike
            savename = filename.replace("_mlp_","_").replace("_zernike.mat", "_ds_zernike.mat")
            io.savemat(savename, {"zernike": zernike.numpy()})
        # load zernike_phase
        trans_dict = io.loadmat(args["phase_zernike_path"].replace("55", str(args["phase_size"])))
        Z2P = torch.from_numpy(trans_dict["Z2P"]).type(torch.float32)
        # plot mean zernike
        Helper.plot_phase_img([mean_zernike, Z2P], cmap="coolwarm", save_name=args["data_path"] + "/sys_abr.png")
        print("System aberration removed! Zernike phase saved as *_ds_zernike.mat.")
    elif args["job"] == "draw":
        # load zernike_phase
        trans_dict = io.loadmat(args["phase_zernike_path"].replace("55", str(args["phase_size"])))
        Z2P = torch.from_numpy(trans_dict["Z2P"]).type(torch.float32)
        # file list
        file_list = glob.glob(args["data_path"] + "/**/*_ds_zernike.mat", recursive=True)
        # plot
        for filename in tqdm(file_list):
            path = "/".join(filename.replace("\\", "/").split("/")[:-1])
            Helper.makedirs(path + "/draw/")
            zernike = torch.from_numpy(io.loadmat(filename)["zernike"]).type(torch.float32)
            Helper.plot_phase_img([zernike[3:,1:-1,1:-1], Z2P[3:]], cmap="coolwarm", caxis=[-525, 525], dpi=150, 
                                  save_name=path + "/draw/" + filename.replace("\\", "/").split("/")[-1].replace("_ds_zernike.mat", "_ds_phase.png"))
        print("Zernike phase drawn! PNG files saved under draw/ folder.")
    else:
        raise Exception("job type error!")



if __name__ == "__main__":
    slope2zernike()