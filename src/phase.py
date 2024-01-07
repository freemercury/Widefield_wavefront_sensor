from helper import *
from phase_detection_model import PhaseProjection
import argparse
from scipy import io


DATA_PATH = "./data/phase_data/230407/set2/"
CKPT_PATH = "./data/log/mlp/"
MASK_PATH = "./data/settings/mask.mat"
JOB = "infer"  # "train" or "test" or "infer"
GPU_ID = 0  # set to None for cpu
MASK_SIZE = [15,15]
HIDDEN_SIZE = [300,500] # hidden layer size of mlp
NUM_ZERNIKE = 35     # number of zernike modes
EPOCH = 10  # number of epochs
LR = 1e-4   # learning rate
BATCH_SIZE = 128    # batch size
VALID = True    # whether to use validation set
SPLIT = 0.9 # train / (train + valid) split ratio

# args
def parse_args():
    parser = argparse.ArgumentParser(description='Phase Projection Arguments')
    parser.add_argument('--data_path', default=DATA_PATH, type=str, help='data path')
    parser.add_argument('--job', default=JOB, type=str, help='job type, "train" or "test" or "infer"')
    parser.add_argument('--ckpt_path', default=CKPT_PATH, type=str, help='checkpoint path')
    parser.add_argument('--mask_path', default=MASK_PATH, type=str, help='mask path')
    parser.add_argument('--gpu_id', default=GPU_ID, type=int, help='gpu id, -1 for cpu')
    parser.add_argument('--mask_size', default=MASK_SIZE, type=str, help='mask size (h,w)')
    parser.add_argument('--hidden_size', default=HIDDEN_SIZE, type=str, help='hidden layer size of mlp')
    parser.add_argument('--num_zernike', default=NUM_ZERNIKE, type=int, help='number of zernike modes')
    parser.add_argument('--epoch', default=EPOCH, type=int, help='number of epochs')
    parser.add_argument('--lr', default=LR, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')
    parser.add_argument('--valid', default=VALID, type=bool, help='whether to use validation set')
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
    Helper.save_json(args, args["data_path"] + "phase_args_" + Helper.get_timestamp() + ".json")
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
    else:
        raise Exception("job type error!")



if __name__ == "__main__":
    slope2zernike()